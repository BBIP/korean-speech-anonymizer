import argparse
import difflib
import json
import os
import re
import tempfile
import time
from pathlib import Path


# 익명화 LLM 모델 경로 및 입출력 폴더 기본값
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/모델주소/gemma-4-31B-it",
)
SERVER_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", str(SERVER_DIR / "Text"))
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", str(SERVER_DIR / "Anonymization"))
INFO_FOLDER = os.environ.get("INFO_FOLDER", str(SERVER_DIR / "Anonymization_info"))
ANONYMIZATION_REGISTRY_FILE = os.environ.get(
    "ANONYMIZATION_REGISTRY_FILE", str(SERVER_DIR / "anonymization_registry.json")
)

# 토큰 및 추론 시간 제한 설정
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "8192"))
OUTPUT_TOKEN_MARGIN = int(os.environ.get("OUTPUT_TOKEN_MARGIN", "256"))
OUTPUT_TOKEN_MARGIN_RATIO = float(
    os.environ.get("OUTPUT_TOKEN_MARGIN_RATIO", "0.5")
)
MAX_NEW_TOKENS = int(
    os.environ.get(
        "MAX_NEW_TOKENS",
        str(
            MAX_INPUT_TOKENS
            + max(
                OUTPUT_TOKEN_MARGIN,
                int(MAX_INPUT_TOKENS * OUTPUT_TOKEN_MARGIN_RATIO),
            )
        ),
    )
)
MAX_INFERENCE_SECONDS = float(os.environ.get("MAX_INFERENCE_SECONDS", "600"))
INFERENCE_TOKENS_PER_SECOND = float(
    os.environ.get("INFERENCE_TOKENS_PER_SECOND", "20")
)
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.0"))

# 끊긴 결과의 꼬리를 로그에 남길 길이. 원문이 찍히므로 파일 로그에서는 0으로 끈다.
TRUNCATED_TAIL_CHARS = int(os.environ.get("TRUNCATED_TAIL_CHARS", "300"))

# 평문·화자분리용 프롬프트. 공통 치환 규칙은 두 쪽을 함께 고쳐야 한다.
PLAIN_SYSTEM_PROMPT = """당신은 한국어 텍스트 익명화 도구입니다.
직업이나 역할에 관계없이 모든 사람을 대상으로 입력 텍스트에서 개인정보를 찾아 다음과 같이 정확히 치환하세요.
- 사람 이름: [이름]
- 생년월일 또는 생일: [생년월일]
- 전화번호나 휴대전화 번호: [전화번호]

규칙:
1. 자신이 치환하지 않는 모든 일반 텍스트(단어, 문맥, 맞춤법, 띄어쓰기, 문장부호, 줄바꿈)는 절대로 변경·수정·삭제·추가하지 말고 원본 그대로 유지해야 합니다.
2. 치환은 오직 개인정보 단어만 1:1로 교체하는 것이며, 개인정보 외의 텍스트를 다듬거나 요약하거나 표준어로 교정하는 행위는 엄격히 금지됩니다.
3. [이름] 바로 뒤에 붙은 호칭, 조사, 어미는 절대로 축약하거나 표준어로 고치지 마세요.
   - 호칭 축약 금지: "홍길동 고객님" -> "[이름] 고객님" (O) / "[이름] 님" (X)
   - 호칭 축약 금지: "홍길동 선생님" -> "[이름] 선생님" (O) / "[이름] 님" (X)
   - 구어체 조사 유지: "홍길동이가" -> "[이름]이가" (O) / "[이름]이" (X)
   - 구어체 조사 유지: "홍길동씨한테" -> "[이름]씨한테" (O) / "[이름]에게" (X)
   - 문장부호 유지: "홍길동, 안녕하세요" -> "[이름], 안녕하세요" (O) / "[이름] 안녕하세요" (X)
4. 이름 한 글자씩 부르는 호칭("홍 길자 동자", "홍 자 길 자 동 자", "김 자 철자 수자" 등)은 '자'를 포함한 이름 호칭 전체를 하나의 [이름]으로 치환합니다.
   - "홍 길자 동자 님" -> "[이름] 님" (O) / "[이름] 동자 님" (X)
5. 추측할 수 있는 이름도 문맥상 사람 이름이면 치환합니다.
6. 이미 치환된 표시는 변경하지 않습니다.
7. 설명, 인사말, 따옴표, 마크다운 없이 익명화된 본문만 출력합니다.
"""

DIARIZED_SYSTEM_PROMPT = """당신은 한국어 대화 전사 익명화 도구입니다.
입력은 한 줄에 한 발화가 담긴 화자분리 전사이며, 모든 줄이 "화자1: 발화" 형식으로 화자 표시와 콜론으로 시작합니다.
직업이나 역할에 관계없이 모든 화자의 발화에서 개인정보를 찾아 다음과 같이 정확히 치환하세요.
- 사람 이름: [이름]
- 생년월일 또는 생일: [생년월일]
- 전화번호나 휴대전화 번호: [전화번호]

규칙:
1. 줄 앞머리의 화자 표시("화자1:", "화자2:", "화자미상:" 등)는 사람 이름이 아닙니다. 절대로 치환하지 말고 콜론과 뒤의 공백까지 원본 그대로 두세요.
   - "화자1: 홍길동입니다" -> "화자1: [이름]입니다" (O) / "[이름]: [이름]입니다" (X)
   - 화자 표시는 이름으로 바꾸지도, 번호를 고치지도, 새로 만들지도 않습니다.
2. 줄 구조를 그대로 유지합니다. 입력과 출력의 줄 수가 같아야 하며, 줄을 합치거나 나누거나 순서를 바꾸거나 빈 줄을 넣지 마세요.
3. 발화 안에서 개인정보가 아닌 모든 텍스트(단어, 문맥, 맞춤법, 띄어쓰기, 문장부호, 줄바꿈)는 절대로 변경·수정·삭제·추가하지 말고 원본 그대로 유지해야 합니다. 말이 끊긴 구어체 발화도 그대로 둡니다.
4. 치환은 오직 개인정보 단어만 1:1로 교체하는 것이며, 개인정보 외의 텍스트를 다듬거나 요약하거나 표준어로 교정하는 행위는 엄격히 금지됩니다.
5. [이름] 바로 뒤에 붙은 호칭, 조사, 어미는 절대로 축약하거나 표준어로 고치지 마세요.
   - 호칭 축약 금지: "홍길동 고객님" -> "[이름] 고객님" (O) / "[이름] 님" (X)
   - 호칭 축약 금지: "홍길동 선생님" -> "[이름] 선생님" (O) / "[이름] 님" (X)
   - 구어체 조사 유지: "홍길동이가" -> "[이름]이가" (O) / "[이름]이" (X)
   - 구어체 조사 유지: "홍길동씨한테" -> "[이름]씨한테" (O) / "[이름]에게" (X)
   - 문장부호 유지: "홍길동, 안녕하세요" -> "[이름], 안녕하세요" (O) / "[이름] 안녕하세요" (X)
6. 이름 한 글자씩 부르는 호칭("홍 길자 동자", "홍 자 길 자 동 자", "김 자 철자 수자" 등)은 '자'를 포함한 이름 호칭 전체를 하나의 [이름]으로 치환합니다.
   - "홍 길자 동자 님" -> "[이름] 님" (O) / "[이름] 동자 님" (X)
   - 한 발화가 여러 줄에 걸쳐 이름을 한 글자씩 부르더라도 줄을 합치지 말고 각 줄에서 나온 부분만 치환합니다.
7. 추측할 수 있는 이름도 문맥상 사람 이름이면 치환합니다.
8. 이미 치환된 표시는 변경하지 않습니다.
9. 설명, 인사말, 따옴표, 마크다운 없이 익명화된 본문만 출력합니다.
"""

# --input-kind 값과 프롬프트를 잇는 표다. 기본은 평문(Text/)이다.
PLAIN_INPUT_KIND = "plain"
DIARIZED_INPUT_KIND = "diarized"
SYSTEM_PROMPTS = {
    PLAIN_INPUT_KIND: PLAIN_SYSTEM_PROMPT,
    DIARIZED_INPUT_KIND: DIARIZED_SYSTEM_PROMPT,
}
INPUT_KINDS = tuple(SYSTEM_PROMPTS)
INPUT_KIND_LABELS = {
    PLAIN_INPUT_KIND: "평문 전사용 프롬프트",
    DIARIZED_INPUT_KIND: "화자분리 전사용 프롬프트",
}

ALL_TYPE_TO_TAG = {
    "name": "[이름]",
    "birthdate": "[생년월일]",
    "phone": "[전화번호]",
}
PLACEHOLDER_TYPES = {
    placeholder: entity_type
    for entity_type, placeholder in ALL_TYPE_TO_TAG.items()
}
PLACEHOLDERS = tuple(PLACEHOLDER_TYPES)

UNRESOLVED_ADJACENT = "치환 표시가 연달아 붙어 경계를 나눌 수 없음"
UNRESOLVED_ANCHOR_LOST = "표시 밖 문장이 원문과 달라 위치 추적이 끊김"
UNRESOLVED_EMPTY = "표시에 대응하는 원문 구간이 비어 있음"

PLACEHOLDER_PATTERN = re.compile(
    "(" + "|".join(re.escape(placeholder) for placeholder in PLACEHOLDERS) + ")"
)

torch = None
tokenizer = None
model = None
DEVICE = "not-loaded"
TORCH_DTYPE = "not-loaded"
STOP_TOKEN_IDS = []


def describe_input_kind(input_kind):
    """입력 종류를 사람이 읽는 프롬프트 이름으로 바꾼다."""
    return INPUT_KIND_LABELS.get(input_kind, "알 수 없는 종류")


def atomic_write_text(path, content, encoding="utf-8"):
    """같은 폴더의 고유 임시 파일을 fsync한 뒤 목적지로 원자 교체한다."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"파일을 원자적으로 저장할 수 없습니다: {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def normalize_phone_digits(value):
    """전화번호 비교용으로 숫자만 남긴다."""
    return "".join(character for character in str(value) if character.isdigit())


def load_anonymization_registry(path=ANONYMIZATION_REGISTRY_FILE):
    """빈 목록을 허용하는 사전 치환용 이름·전화번호 JSON을 읽는다."""
    roster_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(roster_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"익명화 등록 파일을 읽을 수 없습니다: {roster_path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("익명화 등록 파일은 schema_version이 1인 객체여야 합니다.")

    names = document.get("names")
    phones = document.get("phone_numbers")
    if not isinstance(names, list) or not isinstance(phones, list):
        raise RuntimeError("익명화 등록 파일에는 names와 phone_numbers 배열이 필요합니다.")
    for field, values in (("names", names), ("phone_numbers", phones)):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise RuntimeError(f"{field} 항목은 비어 있지 않은 문자열이어야 합니다.")

    names = tuple(dict.fromkeys(name.strip() for name in names))
    if any(placeholder in name for name in names for placeholder in PLACEHOLDERS):
        raise RuntimeError("등록 이름에는 치환 표시를 포함할 수 없습니다.")
    phone_numbers = {}
    for value in phones:
        original = value.strip()
        if not shape_matches("[전화번호]", original):
            raise RuntimeError("등록 전화번호는 7~15자리 숫자와 전화번호 구분자로 구성해야 합니다.")
        phone_numbers.setdefault(normalize_phone_digits(original), original)

    print(f"익명화 등록 목록 로드: 이름 {len(names)}개, 전화번호 {len(phone_numbers)}개")
    return {"source_path": roster_path, "names": names, "phone_numbers": phone_numbers}


def phone_number_pattern(digits):
    """하이픈·공백·점·괄호 표기가 달라도 같은 전체 전화번호를 찾는다."""
    separator = r"[ \t().-]*"
    return re.compile(
        r"(?<!\d)" + separator.join(re.escape(digit) for digit in digits) + r"(?!\d)"
    )


def mask_known_info(text, registry):
    """등록된 이름·전화번호를 표준 치환 표시로 먼저 바꾼다."""
    masked = text
    for digits in sorted(registry["phone_numbers"], key=len, reverse=True):
        masked = phone_number_pattern(digits).sub("[전화번호]", masked)
    for name in sorted(registry["names"], key=len, reverse=True):
        masked = "".join(
            part if part in PLACEHOLDERS else part.replace(name, "[이름]")
            for part in PLACEHOLDER_PATTERN.split(masked)
        )
    return masked


def read_utf8_text(path):
    """개행을 변환하지 않고 UTF-8 텍스트를 읽는다."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def resolve_stop_token_ids():
    """모델 config 및 tokenizer에서 생성 정지 토큰 ID 목록을 추출한다."""
    stop_ids = set()

    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        stop_ids.add(configured)
    elif configured:
        stop_ids.update(configured)

    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    for token in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_ids.add(token_id)

    if not stop_ids:
        raise RuntimeError(
            "정지 토큰을 찾을 수 없습니다. 이 모델로는 생성이 끝나지 않습니다."
        )
    return sorted(stop_ids)


def load_model():
    """로컬 LLM 및 토크나이저를 로드한다."""
    global torch, tokenizer, model, DEVICE, TORCH_DTYPE, STOP_TOKEN_IDS

    if model is not None:
        return

    model_path = Path(MODEL_PATH)
    if not (model_path / "config.json").exists():
        raise RuntimeError(
            f"올바른 모델 폴더가 아닙니다: {MODEL_PATH}\n"
            "config.json이 들어 있는 모델 또는 snapshot 폴더를 지정하세요."
        )

    try:
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch와 transformers가 필요합니다. 서버 가상환경에 설치하세요."
        ) from exc

    torch = torch_module
    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda:0" if use_cuda else "cpu"
    if use_cuda and torch.cuda.is_bf16_supported():
        TORCH_DTYPE = torch.bfloat16
    elif use_cuda:
        TORCH_DTYPE = torch.float16
    else:
        TORCH_DTYPE = torch.float32

    print(f"Loading anonymization LLM from: {MODEL_PATH}")
    print(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    STOP_TOKEN_IDS = resolve_stop_token_ids()
    print(
        "정지 토큰: "
        + ", ".join(
            f"{tokenizer.convert_ids_to_tokens(token_id)}({token_id})"
            for token_id in STOP_TOKEN_IDS
        )
    )


def build_prompt(text, system_prompt=PLAIN_SYSTEM_PROMPT):
    """chat template을 적용하여 시스템/유저 프롬프트를 구성한다."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "이 모델의 chat template을 사용할 수 없습니다. "
            "Instruction 모델인지 확인하세요."
        ) from exc


def calculate_generation_limit(source_token_count):
    """생성 토큰 상한을 원문 길이에 비례해 잡는다(여유분은 고정값과 비례분 중 큰 쪽)."""
    margin = max(
        OUTPUT_TOKEN_MARGIN,
        int(source_token_count * OUTPUT_TOKEN_MARGIN_RATIO),
    )
    return min(
        MAX_NEW_TOKENS,
        max(128, source_token_count + margin),
    )


def calculate_time_limit(generation_limit):
    """생성 상한을 다 뽑을 만큼의 시간을 잡는다(고정 상한은 하한선으로만 쓴다)."""
    if INFERENCE_TOKENS_PER_SECOND <= 0:
        return MAX_INFERENCE_SECONDS
    return max(
        MAX_INFERENCE_SECONDS,
        generation_limit / INFERENCE_TOKENS_PER_SECOND,
    )


def run_inference(text, system_prompt=PLAIN_SYSTEM_PROMPT):
    """로드된 LLM으로 텍스트 한 건을 익명화한다."""
    if model is None or tokenizer is None or torch is None:
        raise RuntimeError("LLM이 아직 로드되지 않았습니다.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text는 비어 있지 않은 문자열이어야 합니다.")

    prompt = build_prompt(text, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[-1]
    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(
            f"입력이 너무 깁니다: {input_length} 토큰 "
            f"(최대 {MAX_INPUT_TOKENS} 토큰)"
        )
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}

    source_token_count = len(
        tokenizer.encode(text, add_special_tokens=False)
    )
    generation_limit = calculate_generation_limit(source_token_count)
    time_limit = calculate_time_limit(generation_limit)
    started_at = time.monotonic()
    print(
        "LLM 추론 시작: "
        f"입력 {source_token_count} 토큰, 생성 상한 {generation_limit} 토큰, "
        f"시간 상한 {time_limit:.0f}초"
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_limit,
            max_time=time_limit,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=STOP_TOKEN_IDS,
        )

    generated_ids = output_ids[0][input_length:]
    generated_count = int(generated_ids.shape[-1])
    # 정지 토큰으로 끝나지 않았다면 상한에 걸려 문서 중간에서 끊긴 결과다.
    stopped_cleanly = (
        generated_count > 0 and int(generated_ids[-1]) in STOP_TOKEN_IDS
    )

    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not result:
        raise RuntimeError("LLM이 빈 익명화 결과를 반환했습니다.")

    elapsed = time.monotonic() - started_at
    if not stopped_cleanly:
        reason = (
            f"생성 토큰 상한 {generation_limit}"
            if generated_count >= generation_limit
            else f"추론 시간 상한 {time_limit:.0f}초"
        )
        # 꼬리를 보면 여유분 부족인지 그리디 디코딩의 반복 루프인지 갈린다.
        if TRUNCATED_TAIL_CHARS > 0:
            print(
                f"  끊긴 결과의 마지막 {TRUNCATED_TAIL_CHARS}자(원문 내용 포함): "
                f"...{result[-TRUNCATED_TAIL_CHARS:]}"
            )
        raise RuntimeError(
            f"익명화가 중간에 끊겼습니다({reason}, {elapsed:.1f}초). "
            "뒷부분이 누락되므로 결과를 저장하지 않습니다."
        )

    print(f"LLM 추론 완료: {elapsed:.1f}초, 생성 {generated_count} 토큰")
    return result


def align_lines(source_lines, anonymized_lines):
    """원문 줄과 익명화 줄을 (원문 줄번호, 원문 줄, 익명화 줄) 목록으로 정렬 매칭한다."""
    if len(source_lines) == len(anonymized_lines):
        return [
            (line_number, source_line, anonymized_line)
            for line_number, (source_line, anonymized_line) in enumerate(
                zip(source_lines, anonymized_lines), start=1
            )
            if source_line != anonymized_line
        ]

    pairs = []
    matcher = difflib.SequenceMatcher(
        None, source_lines, anonymized_lines, autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_block = source_lines[i1:i2]
        anonymized_block = anonymized_lines[j1:j2]
        for offset in range(max(len(source_block), len(anonymized_block))):
            pairs.append(
                (
                    i1 + offset + 1,
                    source_block[offset] if offset < len(source_block) else "",
                    anonymized_block[offset]
                    if offset < len(anonymized_block)
                    else "",
                )
            )
    return pairs


def resolve_known_adjacent_replacements(
    placeholders, region, registry
):
    """등록된 이름·전화번호로 경계가 확정되는 연속 치환 표시의 원문 값을 복원한다."""
    if registry is None or not placeholders or not region:
        return None
    known_tags = {"[이름]", "[전화번호]"}
    if any(placeholder not in known_tags for placeholder in placeholders):
        return None

    known_names = tuple(registry["names"])
    known_numbers = tuple(registry["phone_numbers"])
    solutions = set()

    def walk(index, cursor, values):
        if index == len(placeholders):
            if cursor == len(region):
                solutions.add(tuple(values))
            return

        placeholder = placeholders[index]
        if placeholder == "[이름]":
            for name in known_names:
                if region.startswith(name, cursor):
                    walk(index + 1, cursor + len(name), values + [name])
            return

        separator = r"[ \t().-]*"
        for digits in known_numbers:
            pattern = re.compile(
                separator.join(re.escape(digit) for digit in digits)
            )
            match = pattern.match(region, cursor)
            if match is not None:
                walk(
                    index + 1,
                    match.end(),
                    values + [region[cursor : match.end()]],
                )

    walk(0, 0, [])
    if len(solutions) != 1:
        return None
    values = next(iter(solutions))
    return [
        (placeholder, value, None)
        for placeholder, value in zip(placeholders, values)
    ]


def shape_matches(placeholder, value):
    """원문 조각이 그 치환 표시의 종류로서 말이 되는 형태인지 검사한다.

    인접한 표시의 경계를 나눌 때만 쓴다.
    """
    if not value:
        return False
    if placeholder == "[전화번호]":
        digits = normalize_phone_digits(value)
        return (
            7 <= len(digits) <= 15
            and re.fullmatch(r"\+?[0-9][0-9 \t().-]*[0-9]", value) is not None
        )
    if placeholder == "[생년월일]":
        return (
            len(value) <= 40
            and len(normalize_phone_digits(value)) >= 4
            and re.fullmatch(r"[0-9][0-9년월일생 \t./-]*[0-9일생]", value) is not None
        )
    if placeholder == "[이름]":
        # 이름은 형태 제약이 약해 느슨하게 잡고, 경계는 전화번호·생년월일이 확정한다.
        return (
            len(value) <= 20
            and not any(character.isdigit() for character in value)
            and re.search(r"[,.:;!?/()\[\]{}]", value) is None
        )
    return False


def diff_value_is_plausible(placeholder, value):
    """diff로 되짚은 값이 지워진 개인정보로서 말이 되는지 엄격히 검사한다.

    앵커 탐색과 달리 diff에는 경계 보장이 없어, 함께 지워진 옆 문장이 값으로
    빨려 들어갈 수 있다. 그런 값은 기록에 없느니만 못하므로 거부한다.
    """
    if not value or value != value.strip():
        return False
    if placeholder == "[전화번호]":
        return shape_matches("[전화번호]", value)
    if placeholder == "[생년월일]":
        return shape_matches("[생년월일]", value)
    # 이름은 띄어쓰기로 본다. 실제 이름은 붙여 쓰고, 한 글자씩 부르는 호칭
    # ("홍 길자 동자")도 조각이 한두 글자다. 세 글자를 넘으면 문장이 섞인 것이다.
    if any(character.isdigit() for character in value):
        return False
    if re.search(r"[,.:;!?/()\[\]{}]", value):
        return False
    pieces = value.split()
    if len(pieces) > 1 and any(len(piece) > 2 for piece in pieces):
        return False
    return len(value) <= 20


def resolve_adjacent_by_shape(placeholders, region):
    """종류별 형태 규칙으로 인접 치환의 경계가 유일할 때만 나눈다."""
    if not placeholders or not region:
        return None

    solutions = []

    def walk(index, cursor, values):
        if len(solutions) > 1:
            return
        if index == len(placeholders):
            if cursor == len(region):
                solutions.append(tuple(values))
            return
        # 남은 표시마다 최소 한 글자는 있어야 하므로 그만큼 끝을 남긴다.
        remaining = len(placeholders) - index - 1
        for end in range(cursor + 1, len(region) - remaining + 1):
            piece = region[cursor:end]
            if shape_matches(placeholders[index], piece):
                walk(index + 1, end, values + [piece])

    walk(0, 0, [])
    if len(solutions) != 1:
        return None
    return [
        (placeholder, value, None)
        for placeholder, value in zip(placeholders, solutions[0])
    ]


def resolve_adjacent_replacements(placeholders, region, registry=None):
    """붙어 있는 치환 표시들의 원문 값을 확정한다.

    등록 목록 대조를 먼저, 실패하면 형태 규칙으로 나눈다. 경계가 유일할 때만
    값을 돌려준다.
    """
    known = resolve_known_adjacent_replacements(
        placeholders, region, registry
    )
    if known is not None:
        return known
    return resolve_adjacent_by_shape(placeholders, region)


def tokenize_line(line):
    """치환 표시는 한 덩어리로, 나머지는 한 글자씩인 토큰 목록을 만든다.

    표시를 원자 단위로 묶지 않으면 `[이름]` 안의 '이름'이 원문의 '이름은' 같은
    평범한 문장에 매칭되어 표시가 쪼개진다.
    """
    tokens = []
    for index, part in enumerate(PLACEHOLDER_PATTERN.split(line)):
        if index % 2 == 1:
            tokens.append(part)
        else:
            tokens.extend(part)
    return tokens


def extract_replacements_by_diff(
    source_line, anonymized_line, registry=None
):
    """표시 밖 문장까지 달라진 줄을 diff로 되짚는다. 앵커 탐색이 끊겼을 때만 쓴다.

    변경 블록의 결과 쪽이 치환 표시로만 이루어져 있으면 원문 쪽 전체가 지워진
    값이다. 표시 밖 문자가 섞인 블록은 경계를 알 수 없어 확정하지 않는다.
    """
    source_tokens = tokenize_line(source_line)
    result_tokens = tokenize_line(anonymized_line)
    matcher = difflib.SequenceMatcher(
        None, source_tokens, result_tokens, autojunk=False
    )
    replacements = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_part = "".join(source_tokens[i1:i2])
        result_part = "".join(result_tokens[j1:j2])
        placeholders = tuple(
            token for token in result_tokens[j1:j2] if token in PLACEHOLDER_TYPES
        )
        if not placeholders:
            # 개인정보와 무관한 편집(맞춤법·띄어쓰기 등)이다.
            continue
        if "".join(placeholders) != result_part:
            replacements.extend(
                (placeholder, None, UNRESOLVED_ANCHOR_LOST)
                for placeholder in placeholders
            )
            continue
        # "홍길동 님"을 "[이름]님"으로 쓸 때처럼 원문 쪽에 딸려 온 양끝 공백을 떼어 낸다.
        source_part = source_part.strip()
        if not source_part:
            replacements.extend(
                (placeholder, None, UNRESOLVED_EMPTY)
                for placeholder in placeholders
            )
            continue
        if len(placeholders) == 1:
            if diff_value_is_plausible(placeholders[0], source_part):
                replacements.append((placeholders[0], source_part, None))
            else:
                replacements.append(
                    (placeholders[0], None, UNRESOLVED_ANCHOR_LOST)
                )
            continue
        resolved = resolve_adjacent_replacements(
            placeholders, source_part, registry
        )
        if resolved is not None and all(
            diff_value_is_plausible(placeholder, value)
            for placeholder, value, _ in resolved
        ):
            replacements.extend(resolved)
        else:
            replacements.extend(
                (placeholder, None, UNRESOLVED_ADJACENT)
                for placeholder in placeholders
            )
    return replacements


def extract_replacements(
    source_line, anonymized_line, registry=None
):
    """치환 표시를 기준으로 원문의 어떤 표현이 변경되었는지 역추적한다."""
    parts = PLACEHOLDER_PATTERN.split(anonymized_line)

    replacements = []
    cursor = 0
    pending = []

    def resolve(region, lost_anchor=False):
        # "홍길동 님"을 "[이름]님"으로 쓸 때처럼 앵커 앞 구간에 딸려 온 양끝 공백을 떼어 낸다.
        if region:
            region = region.strip()
        if len(pending) == 1:
            if region:
                replacements.append((pending[0], region, None))
            else:
                reason = UNRESOLVED_ANCHOR_LOST if lost_anchor else UNRESOLVED_EMPTY
                replacements.append((pending[0], None, reason))
        else:
            resolved = (
                None
                if lost_anchor
                else resolve_adjacent_replacements(
                    tuple(pending), region, registry
                )
            )
            if resolved is not None:
                replacements.extend(resolved)
            else:
                reason = (
                    UNRESOLVED_ANCHOR_LOST
                    if lost_anchor
                    else UNRESOLVED_ADJACENT
                )
                replacements.extend(
                    (placeholder, None, reason) for placeholder in pending
                )
        pending.clear()

    for index, part in enumerate(parts):
        if index % 2 == 1:
            pending.append(part)
            continue

        if not part:
            continue

        found = source_line.find(part, cursor)
        if found < 0:
            # 앵커가 끊겼으므로 여기까지를 버리고 줄 전체를 diff로 다시 되짚는다.
            return extract_replacements_by_diff(
                source_line, anonymized_line, registry
            )

        if pending:
            resolve(source_line[cursor:found])
        cursor = found + len(part)

    if pending:
        resolve(source_line[cursor:])
    return replacements


def summarize_redactions(source_text, anonymized_text, registry=None):
    """원문과 익명화 결과를 비교하여 치환 내역 요약 정보를 생성한다."""
    source_lines = source_text.splitlines()
    anonymized_lines = anonymized_text.splitlines()

    entries = []
    total = 0
    for line_number, source_line, anonymized_line in align_lines(
        source_lines, anonymized_lines
    ):
        for tag, value, reason in extract_replacements(
            source_line, anonymized_line, registry
        ):
            # 원문에 이미 있던 표시는 이번에 지운 것이 아니므로 세지 않는다.
            if value in PLACEHOLDERS:
                continue
            # 같은 줄·값·사유가 반복되면 한 항목으로 묶고 건수만 올린다.
            if (
                entries
                and entries[-1]["line"] == line_number
                and entries[-1]["tag"] == tag
                and entries[-1]["value"] == value
                and entries[-1]["reason"] == reason
            ):
                entries[-1]["count"] += 1
            else:
                entry = {
                    "line": line_number,
                    "tag": tag,
                    "type": PLACEHOLDER_TYPES[tag],
                    "value": value,
                    "resolved": value is not None,
                    "reason": reason,
                    "count": 1,
                }
                entries.append(entry)
            total += 1

    return {
        "entries": entries,
        "total": total,
        "unresolved": sum(
            entry["count"] for entry in entries if not entry["resolved"]
        ),
    }


def build_info_document(summary, file_label):
    """모든 참여자의 치환 값을 이름·생년월일·전화번호별로 기록한다."""
    grouped = {entity_type: [] for entity_type in ALL_TYPE_TO_TAG}
    for entry in summary["entries"]:
        values = grouped[entry["type"]]
        if entry["value"] not in values:
            values.append(entry["value"])
    return {"file": file_label, "info": grouped}


def render_info_json(summary, file_label):
    document = build_info_document(summary, file_label)
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def find_txt_files(input_dir):
    """입력 폴더 아래의 TXT 파일을 경로순으로 반환한다."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    )


def limit_sources(sources, sample, unit):
    """샘플 실행용으로 경로순 상위 sample개를 반환한다."""
    if sample is None:
        return sources
    if sample < 1:
        raise ValueError(f"샘플 개수는 1 이상이어야 합니다: {sample}")

    total = len(sources)
    if sample >= total:
        print(f"샘플 {sample}개를 요청했지만 전체가 {total}개라 전부 처리합니다.")
        return sources

    print(f"샘플 실행: 전체 {total}개 중 경로순 앞에서 {sample}개만 처리합니다.")
    print(f"  (전체를 돌리려면 --sample 없이 실행하세요. 단위: {unit})")
    return sources[:sample]


def parse_shard(value):
    """`--shard I/N`을 (몫 번호, 전체 몫 수)로 해석한다.

    argparse의 type=으로 넘기면 아래 오류 문구가 삼켜지므로 parse_args()에서 부른다.
    """
    parts = str(value).strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard 형식은 I/N입니다: {value}")
    try:
        index, count = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--shard의 두 값은 정수여야 합니다: {value}") from exc
    if count < 1:
        raise ValueError(f"--shard의 N은 1 이상이어야 합니다: {value}")
    if not 0 <= index < count:
        raise ValueError(
            f"--shard의 I는 0 이상 {count} 미만이어야 합니다: {value}"
        )
    return index, count


def shard_sources(sources, shard, unit):
    """이 프로세스가 맡을 몫만 남긴다.

    연속 블록이 아니라 한 칸씩 건너뛰며(`sources[I::N]`) 나눈다. 경로순 목록에는
    길이가 비슷한 전사가 몰려 있어 블록으로 자르면 부하가 한쪽에 쏠린다.
    """
    if shard is None:
        return sources

    index, count = shard
    if count == 1:
        return sources

    selected = sources[index::count]
    print(
        f"분산 실행: 전체 {len(sources)}개 중 샤드 {index}/{count}의 몫 "
        f"{len(selected)}개를 처리합니다. (단위: {unit})"
    )
    if not selected:
        print("  이 샤드가 맡을 파일이 없습니다. 샤드 수가 파일 수보다 많습니다.")
    return selected


def build_jobs(
    input_folder,
    output_folder,
    info_folder=INFO_FOLDER,
    sample=None,
    shard=None,
):
    """하위 구조를 유지하는 (입력 TXT, 출력 TXT, 기록 JSON) 작업 목록을 생성한다."""
    input_dir = Path(input_folder).expanduser().resolve()
    output_dir = Path(output_folder).expanduser().resolve()
    info_dir = Path(info_folder).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"입력 경로는 폴더여야 합니다: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"출력 경로는 폴더여야 합니다: {output_dir}")
    if info_dir.exists() and not info_dir.is_dir():
        raise ValueError(f"기록 경로는 폴더여야 합니다: {info_dir}")

    txt_files = find_txt_files(input_dir)
    if not txt_files:
        raise ValueError(f"입력 폴더에 TXT 파일이 없습니다: {input_dir}")

    # 샘플이 먼저다. 순서를 바꾸면 샤드마다 --sample개씩 처리하게 된다.
    txt_files = limit_sources(txt_files, sample, "TXT 파일")
    txt_files = shard_sources(txt_files, shard, "TXT 파일")

    return [
        (
            source,
            output_dir / source.relative_to(input_dir),
            (info_dir / source.relative_to(input_dir)).with_suffix(".json"),
        )
        for source in txt_files
    ]


def reusable_text_output(source_path, output_path):
    """--resume 옵션 시 재사용 가능한 기존 익명화 TXT가 있는지 확인한다."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    try:
        if output_path.name != source_path.name:
            return False
        if not output_path.is_file():
            return False
        return bool(output_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def write_run_summary(path, summary):
    """run_pipeline.py가 읽을 단계 집계를 JSON으로 저장한다(경로·사유는 제외)."""
    atomic_write_text(
        path,
        json.dumps(
            {
                "total": summary["total"],
                "succeeded": summary["succeeded"],
                "skipped": summary["skipped"],
                "failed": summary["failed"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def delete_failed_result(output_path):
    """실패한 파일의 이전 결과를 지운다. 삭제한 경로 또는 None을 반환한다."""
    output_path = Path(output_path)
    if not output_path.is_file():
        return None
    output_path.unlink()
    return output_path


def process_folder(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    info_folder=INFO_FOLDER,
    sample=None,
    shard=None,
    resume=False,
    registry=None,
    ensure_model_loaded=None,
    input_kind=PLAIN_INPUT_KIND,
    write_info=True,
):
    """입력 폴더 내 TXT 파일을 순회 익명화하고 결과 및 기록을 저장한다.

    등록된 이름·전화번호는 추론 전에 치환한다. input_kind는 시스템 프롬프트를
    고르고("plain"/"diarized"), write_info가 False면 기록 JSON을 쓰지 않는다.
    """
    if input_kind not in SYSTEM_PROMPTS:
        raise ValueError(
            f"입력 종류는 {', '.join(INPUT_KINDS)} 중 하나여야 합니다: {input_kind}"
        )
    system_prompt = SYSTEM_PROMPTS[input_kind]

    if registry is None:
        registry = load_anonymization_registry()

    jobs = build_jobs(input_folder, output_folder, info_folder, sample, shard)
    # 기록 JSON의 file 값은 이 폴더 기준 상대 경로다.
    input_dir = Path(input_folder).expanduser().resolve()

    total = len(jobs)
    succeeded = 0
    skipped = 0
    failures = []

    for index, (source_path, output_path, info_path) in enumerate(jobs, start=1):
        relative = source_path.relative_to(input_dir)
        reuse_result = resume and reusable_text_output(source_path, output_path)
        reprocessing = not reuse_result
        if reuse_result:
            print(f"[{index}/{total}] 기존 익명화 TXT 사용: {output_path}")
        else:
            if ensure_model_loaded is not None:
                ensure_model_loaded()
            print(f"[{index}/{total}] 익명화 중: {source_path}")

        try:
            source_text = read_utf8_text(source_path)
            result = (
                read_utf8_text(output_path).strip()
                if reuse_result
                else run_inference(
                    mask_known_info(source_text, registry),
                    system_prompt,
                )
            )
            # 원문과 결과를 대조해 역추적한다. 표시 밖 문장이 달라진 항목은
            # 원문 값을 확정하지 못하고 unresolved로 남는다.
            summary = summarize_redactions(
                source_text,
                result,
                registry,
            )
            report_payload = (
                render_info_json(summary, relative.as_posix())
                if write_info
                else None
            )

            if not reuse_result:
                atomic_write_text(output_path, result + "\n")
            if write_info:
                atomic_write_text(info_path, report_payload)
        except (OSError, RuntimeError, ValueError) as exc:
            deleted = None
            if reprocessing:
                # 남겨 두면 다음 --resume이 예전 result를 성공으로 오해한다.
                try:
                    deleted = delete_failed_result(output_path)
                except OSError as delete_exc:
                    exc = RuntimeError(
                        f"{exc}; 실패한 기존 result를 지우지 못했습니다: "
                        f"{delete_exc}"
                    )
            failures.append((source_path, str(exc)))
            print(f"  실패: {exc}")
            if deleted is not None:
                print(f"  실패 result 삭제: {deleted}")
            continue

        if reuse_result:
            skipped += 1
        else:
            succeeded += 1
            print(f"  저장 완료: {output_path}")
        # 값을 못 되짚은 건수는 기록을 열어 봐야 할 파일이라는 뜻이다.
        unresolved_note = (
            f", 원문 값 확인 불가 {summary['unresolved']}건"
            if summary["unresolved"]
            else ""
        )
        if write_info:
            print(
                f"  기록 저장: {info_path} "
                f"(치환 {summary['total']}건{unresolved_note})"
            )
        else:
            print(
                f"  치환 {summary['total']}건{unresolved_note} "
                "(기록 JSON은 남기지 않습니다)"
            )

    return {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="로컬 LLM으로 TXT 폴더를 일괄 익명화"
    )
    parser.add_argument(
        "--input-folder",
        default=INPUT_FOLDER,
        help="입력 TXT 폴더 (기본값: 실행 파일 폴더/Text)",
    )
    parser.add_argument(
        "--output-folder",
        default=OUTPUT_FOLDER,
        help="익명화 결과 폴더 (기본값: 실행 파일 폴더/Anonymization)",
    )
    parser.add_argument(
        "--info-folder",
        default=INFO_FOLDER,
        help=(
            "익명화 기록 폴더 (기본값: 실행 파일 폴더/Anonymization_info). "
            "--no-info를 주면 쓰지 않습니다"
        ),
    )
    parser.add_argument(
        "--input-kind",
        choices=INPUT_KINDS,
        default=PLAIN_INPUT_KIND,
        help=(
            "입력 텍스트 종류에 맞는 시스템 프롬프트를 고릅니다 "
            f"(기본값: %(default)s). {PLAIN_INPUT_KIND}는 평문 전사(Text/), "
            f"{DIARIZED_INPUT_KIND}는 '화자1: 발화' 형식의 화자분리 "
            "전사(Diarization/)용입니다"
        ),
    )
    parser.add_argument(
        "--no-info",
        dest="write_info",
        action="store_false",
        help=(
            "익명화 기록 JSON을 남기지 않습니다. 치환 건수는 로그에만 남고 "
            "--info-folder는 쓰지 않으므로, 결과 폴더 하나만 만듭니다 "
            "(화자분리 전사처럼 평문 쪽 기록으로 이미 색인된 입력에 씁니다)"
        ),
    )
    parser.add_argument(
        "--registry-file",
        default=ANONYMIZATION_REGISTRY_FILE,
        help=(
            "사전 치환할 이름·전화번호 JSON "
            "(기본값: 실행 파일 폴더/anonymization_registry.json)"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        help=(
            "샘플 실행: 경로순 앞에서 이 개수만큼만 익명화 "
            "(예: --sample 10, 기본값은 전체 익명화)"
        ),
    )
    parser.add_argument(
        "--shard",
        help=(
            "여러 GPU 분산 실행용: I/N 형식으로 이 프로세스가 맡을 몫 "
            "(예: --shard 0/4). 파일을 한 칸씩 건너뛰며 나눕니다. "
            "직접 쓰기보다 run_pipeline.py --gpus를 쓰는 편이 편합니다"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "같은 상대 경로에 비어 있지 않은 익명화 TXT가 이미 있으면 그 파일의 "
            "LLM 추론을 건너뜁니다. 원문과 결과의 내용은 비교하지 않으므로, "
            "다시 익명화하려면 --resume 없이 실행하거나 해당 TXT를 지우세요"
        ),
    )
    parser.add_argument(
        "--summary-json",
        help=(
            "전체·성공·건너뜀·실패 개수를 JSON으로 저장할 경로 "
            "(run_pipeline.py가 단계별 집계를 읽는 데 씁니다)"
        ),
    )

    args = parser.parse_args(argv)

    if args.sample is not None and args.sample < 1:
        parser.error(f"--sample은 1 이상이어야 합니다: {args.sample}")
    if args.shard is not None:
        try:
            args.shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))

    return args


def main(argv=None):
    args = parse_args(argv)

    try:
        registry = load_anonymization_registry(args.registry_file)
    except RuntimeError as exc:
        print(f"오류: {exc}")
        return 2

    print(f"입력 종류: {args.input_kind} ({describe_input_kind(args.input_kind)})")
    if not args.write_info:
        print("기록 JSON을 남기지 않습니다 (--no-info).")

    model_loaded = False

    def ensure_model_loaded():
        nonlocal model_loaded
        if model_loaded:
            return
        load_model()
        model_loaded = True

    try:
        summary = process_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            info_folder=args.info_folder,
            sample=args.sample,
            shard=args.shard,
            resume=args.resume,
            registry=registry,
            ensure_model_loaded=ensure_model_loaded,
            input_kind=args.input_kind,
            write_info=args.write_info,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        return 2

    if args.summary_json:
        try:
            write_run_summary(args.summary_json, summary)
        except (OSError, RuntimeError) as exc:
            # 집계 저장 실패로 정상 처리된 결과를 실패로 만들지는 않는다.
            print(f"경고: 처리 결과 요약을 저장하지 못했습니다: {exc}")

    print("\n처리 결과:")
    print(f"  전체: {summary['total']}개")
    print(f"  성공: {summary['succeeded']}개")
    print(f"  건너뜀: {summary['skipped']}개")
    print(f"  실패: {summary['failed']}개")

    if summary["failures"]:
        print("\n실패한 파일:")
        for source_path, reason in summary["failures"]:
            print(f"  - {source_path}: {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
