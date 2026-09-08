import argparse
import gc
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path


# 모델 경로 및 입출력 폴더 기본값
MODEL_PATH = os.environ.get("MODEL_PATH", "/모델주소/whisper-large-v3")
SERVER_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", str(SERVER_DIR / "Original"))
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", str(SERVER_DIR / "Text"))
STT_CACHE_FOLDER = os.environ.get(
    "STT_CACHE_FOLDER", str(SERVER_DIR / "STT-cache")
)
STT_TIMESTAMPS_FOLDER = os.environ.get(
    "STT_TIMESTAMPS_FOLDER", str(SERVER_DIR / "STT-Timestamps")
)
LANGUAGE = os.environ.get("LANGUAGE", "korean")

# pyannote 모델 폴더, config.yaml 경로 또는 Hugging Face 저장소 ID를 지정한다.
DIARIZATION_MODEL_PATH = os.environ.get(
    "DIARIZATION_MODEL_PATH", "/모델주소/pyannote-community-1"
)

# 외부 청킹은 경계 중복·누락과 청크별 환각을 일으켜 기본으로 끈다(0).
WORD_CHUNK_SECONDS = float(os.environ.get("WORD_CHUNK_SECONDS", "0"))
WORD_CHUNK_STRIDE_SECONDS = float(
    os.environ.get("WORD_CHUNK_STRIDE_SECONDS", "5")
)

SPEAKER_LABEL_FORMAT = os.environ.get("SPEAKER_LABEL_FORMAT", "화자{index}")
UNKNOWN_SPEAKER_LABEL = os.environ.get("UNKNOWN_SPEAKER_LABEL", "화자미상")

# 화자 수 설정 (통화 녹음 기본 2명 고정, 0이면 하한만 적용)
NUM_SPEAKERS = int(os.environ.get("NUM_SPEAKERS", "2"))
MIN_SPEAKERS = int(os.environ.get("MIN_SPEAKERS", "2"))
DEFAULT_NUM_SPEAKERS = NUM_SPEAKERS if NUM_SPEAKERS > 0 else None
DEFAULT_MIN_SPEAKERS = None if DEFAULT_NUM_SPEAKERS else MIN_SPEAKERS

SPEAKER_MERGE_MAX_GAP = float(os.environ.get("SPEAKER_MERGE_MAX_GAP", "1.0"))
SPEAKER_MERGE_MAX_CHARS = int(os.environ.get("SPEAKER_MERGE_MAX_CHARS", "300"))
OVERLAP_TIE_TOLERANCE = float(os.environ.get("OVERLAP_TIE_TOLERANCE", "0.05"))

SPEAKER_SMOOTH_MAX_CHUNKS = int(os.environ.get("SPEAKER_SMOOTH_MAX_CHUNKS", "2"))
SPEAKER_SMOOTH_MAX_SECONDS = float(
    os.environ.get("SPEAKER_SMOOTH_MAX_SECONDS", "0.8")
)
SPEAKER_SMOOTH_MAX_GAP = float(os.environ.get("SPEAKER_SMOOTH_MAX_GAP", "0.05"))

TIMESTAMP_RESET_TOLERANCE = float(
    os.environ.get("TIMESTAMP_RESET_TOLERANCE", "0.5")
)
# 전사 범위가 오디오 길이의 이 비율에 못 미치면 뒷부분 누락을 경고한다.
TIMELINE_COVERAGE_MIN_RATIO = float(
    os.environ.get("TIMELINE_COVERAGE_MIN_RATIO", "0.9")
)

# Whisper 디코딩 및 환각 억제 파라미터
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4)
COMPRESSION_RATIO_THRESHOLD = float(
    os.environ.get("COMPRESSION_RATIO_THRESHOLD", "1.35")
)
LOGPROB_THRESHOLD = float(os.environ.get("LOGPROB_THRESHOLD", "-1.0"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "3"))
# 올리면 무음 창을 건너뛰지 않아 환각이 나온다.
NO_SPEECH_THRESHOLD = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6"))

# pyannote가 감지한 음성 중 Whisper 단어가 덮지 못한 구간을 재전사한다.
MIN_MISSING_SPEECH_SECONDS = float(
    os.environ.get("MIN_MISSING_SPEECH_SECONDS", "1.0")
)
MISSING_SPEECH_RETRY_PADDING_SECONDS = float(
    os.environ.get("MISSING_SPEECH_RETRY_PADDING_SECONDS", "0.5")
)
MISSING_SPEECH_MAX_RETRY_SECONDS = float(
    os.environ.get("MISSING_SPEECH_MAX_RETRY_SECONDS", "20")
)
WORD_COVERAGE_PADDING_SECONDS = float(
    os.environ.get("WORD_COVERAGE_PADDING_SECONDS", "0.15")
)
MISSING_SPEECH_RETRY_LOOPS = int(
    os.environ.get("MISSING_SPEECH_RETRY_LOOPS", "1")
)


torch = None
transcriber = None
diarizer = None
DEVICE = "not-loaded"
TORCH_DTYPE = "not-loaded"


def clear_gpu_cache():
    """임시 텐서 가비지 수거 및 GPU 미사용 예약 캐시를 정리한다."""
    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(whisper_model_path=None):
    """Whisper 모델 및 프로세서를 로드하여 전역 transcriber 파이프라인을 생성한다."""
    global torch, transcriber, DEVICE, TORCH_DTYPE

    if transcriber is not None:
        return

    whisper_model_path = whisper_model_path or MODEL_PATH
    model_path = Path(whisper_model_path).expanduser()
    if not (model_path / "config.json").exists():
        raise RuntimeError(
            f"올바른 Whisper 모델 폴더가 아닙니다: {model_path}\n"
            "config.json이 들어 있는 모델 또는 snapshot 폴더를 지정하세요."
        )

    try:
        import torch as torch_module
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "torch와 transformers가 필요합니다. 서버 가상환경에 설치하세요."
        ) from exc

    torch = torch_module
    use_cuda = torch.cuda.is_available()
    DEVICE = "cuda:0" if use_cuda else "cpu"
    TORCH_DTYPE = torch.float16 if use_cuda else torch.float32

    print(f"Loading Whisper model from: {model_path}")
    print(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )
    print(f"Pipeline device: {transcriber.device}")


def resolve_diarization_source(model_path=None):
    """로컬 config.yaml 경로 또는 Hugging Face 허브 ID를 검증 및 반환한다."""
    raw = model_path if model_path is not None else DIARIZATION_MODEL_PATH
    candidate = Path(raw).expanduser()

    if candidate.is_dir():
        config_path = candidate / "config.yaml"
        if not config_path.exists():
            raise RuntimeError(
                f"config.yaml이 없습니다: {config_path}\n"
                "pyannote 파이프라인 폴더인지 확인하세요."
            )
        return str(config_path)

    if candidate.exists():
        return str(candidate)

    if raw.count("/") == 1 and not raw.startswith((".", "/", "~")):
        return raw

    raise RuntimeError(
        f"화자 분리 모델을 찾을 수 없습니다: {raw}\n"
        "config.yaml이 들어 있는 로컬 폴더나 허브 ID"
        "(예: pyannote/speaker-diarization-community-1)를 지정하세요."
    )


def load_diarizer(model_path=None):
    """pyannote.audio 화자 분리 파이프라인을 로드한다 (None이면 환경변수 경로)."""
    global torch, diarizer

    if diarizer is not None:
        return

    source = resolve_diarization_source(model_path)

    try:
        import torch as torch_module
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio가 필요합니다. 서버 가상환경에 설치하세요: "
            "pip install pyannote.audio"
        ) from exc

    if torch is None:
        torch = torch_module

    print(f"Loading diarization pipeline from: {source}")
    pipeline = Pipeline.from_pretrained(source)
    if pipeline is None:
        raise RuntimeError(
            f"화자 분리 파이프라인을 만들지 못했습니다: {source}\n"
            "로컬 폴더 경로 또는 허브 접근 권한(토큰)을 확인하세요."
        )

    if DEVICE != "not-loaded":
        device = torch.device(DEVICE)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarizer = pipeline
    print(f"Diarization device: {device}")


def run_diarization(
    audio_path,
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
):
    """WAV 파일에서 화자 구간 [(시작초, 끝초, 화자명), ...]을 추출한다."""
    if diarizer is None:
        raise RuntimeError("화자 분리 모델이 아직 로드되지 않았습니다.")

    options = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers

    constraint = (
        ", ".join(f"{name}={value}" for name, value in options.items())
        or "제약 없음"
    )
    print(f"  화자 분리 시작: {constraint}")

    started_at = time.monotonic()
    annotation = extract_annotation(diarizer(str(audio_path), **options))

    turns = sorted(
        (segment.start, segment.end, label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    )
    turns = label_speakers(turns)

    elapsed = time.monotonic() - started_at
    speakers = len({label for _, _, label in turns})
    print(f"  화자 분리 완료: {speakers}명, {len(turns)}구간, {elapsed:.1f}초")
    return turns


def extract_annotation(output):
    """pyannote 결과 객체에서 itertracks를 지원하는 Annotation을 추출한다."""
    if hasattr(output, "itertracks"):
        return output

    for name in ("speaker_diarization", "diarization", "annotation"):
        candidate = getattr(output, name, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate

    raise RuntimeError(
        "화자 분리 결과에서 구간 정보를 찾지 못했습니다: "
        f"{type(output).__name__}"
    )


def label_speakers(turns):
    """원본 화자 ID(SPEAKER_00 등)를 등장 순서대로 화자1, 화자2 등으로 매핑한다."""
    names = {}
    labeled = []

    for start, end, raw_label in turns:
        if raw_label not in names:
            names[raw_label] = SPEAKER_LABEL_FORMAT.format(index=len(names) + 1)
        labeled.append((start, end, names[raw_label]))

    return labeled


def normalize_chunks(raw_chunks, audio_end=None):
    """결측 타임스탬프를 보정해 (시작, 끝, 텍스트)로 정규화한다.

    Whisper가 마지막 조각의 끝을 None으로 남기면 길이 0인 조각이 되어 화자 배정과
    커버리지 계산이 어긋난다. audio_end를 주면 그 값으로 채운다.
    """
    prepared = []
    for chunk in raw_chunks or []:
        text = (chunk.get("text") or "").rstrip()
        if not text.strip():
            continue
        timestamp = chunk.get("timestamp") or (None, None)
        start = timestamp[0] if len(timestamp) > 0 else None
        end = timestamp[1] if len(timestamp) > 1 else None
        prepared.append([start, end, text])

    previous_end = 0.0
    for index, item in enumerate(prepared):
        if item[0] is None:
            item[0] = previous_end
        if item[1] is None:
            next_start = next(
                (
                    later[0]
                    for later in prepared[index + 1:]
                    if later[0] is not None
                ),
                None,
            )
            if next_start is not None and next_start > item[0]:
                item[1] = next_start
            elif audio_end is not None and audio_end > item[0]:
                item[1] = audio_end
            else:
                item[1] = item[0]
        previous_end = item[1]

    return [tuple(item) for item in prepared]


def repair_chunk_timeline(chunks):
    """창별 상대시간으로 인해 역행된 타임스탬프를 절대 시각으로 보정한다."""
    repaired = []
    offset = 0.0
    previous_end = 0.0
    resets = 0

    for start, end, text in chunks:
        if start + offset < previous_end - TIMESTAMP_RESET_TOLERANCE:
            offset = previous_end - start
            resets += 1
        start += offset
        end = max(end + offset, start)
        repaired.append((start, end, text))
        previous_end = end

    return repaired, resets


def read_wav_duration(path):
    """WAV 재생 길이(초)를 읽는다. 헤더 실패 시 ffprobe, 둘 다 실패하면 None."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (OSError, EOFError, wave.Error):
        frames = rate = 0

    if frames > 0 and rate > 0:
        return frames / float(rate)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def transcript_coverage(chunks, audio_seconds):
    """(전사 끝 시각, 커버리지 비율)을 반환한다. 길이를 모르면 비율은 None."""
    transcript_end = max((end for _, end, _ in chunks), default=0.0)
    if not audio_seconds or audio_seconds <= 0:
        return transcript_end, None
    return transcript_end, transcript_end / audio_seconds


def warn_short_transcript(transcript_end, audio_seconds, coverage, folded=None):
    """커버리지가 기준 미달이면 경고를 출력하고 미달 여부를 반환한다.

    folded(repair_chunk_timeline의 보정 횟수)로 타임스탬프 접힘과 본문 누락을
    구분해 안내한다.
    """
    if coverage is None or coverage >= TIMELINE_COVERAGE_MIN_RATIO:
        return False

    print(
        f"  경고: 전사 타임스탬프는 {transcript_end:.1f}초에서 끝나는데 "
        f"오디오는 {audio_seconds:.1f}초입니다"
        f"(커버리지 {coverage:.0%}, 기준 {TIMELINE_COVERAGE_MIN_RATIO:.0%})."
    )
    if folded:
        print(
            f"  → 직전에 타임스탬프 보정이 {folded}번 있었습니다. "
            "창별 상대시간으로 접힌 쪽이 원인일 수 있습니다."
        )
    else:
        print(
            "  → 타임스탬프 보정은 없었습니다. 뒷부분 본문 자체가 누락된 것으로 "
            "보입니다. 무음/저확률로 판정된 창은 텍스트도 로그도 남기지 않고 "
            "건너뛰어집니다(NO_SPEECH_THRESHOLD, LOGPROB_THRESHOLD)."
        )
    return True


def warn_timeline_mismatch(chunks, turns, folded=None):
    """화자 구간의 마지막 끝 시각을 오디오 길이로 삼아 뒷부분 누락을 경고한다.

    화자분리 단계용 진입점이다. pyannote가 뒷부분을 놓치면 기준값도 짧아지므로
    STT 단계의 read_wav_duration 검사가 더 정확하다.
    """
    if not chunks or not turns:
        return False

    audio_end = max(end for _, end, _ in turns)
    transcript_end, coverage = transcript_coverage(chunks, audio_end)
    return warn_short_transcript(transcript_end, audio_end, coverage, folded)


def pick_speaker(start, end, turns):
    """지정된 시간 구간과 가장 많이 겹치는 화자를 선택한다."""
    best_label = None
    best_overlap = 0.0
    best_remainder = 0.0

    for turn_start, turn_end, label in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap <= 0:
            continue

        remainder = turn_end - end

        if best_label is None:
            better = True
        elif overlap > best_overlap + OVERLAP_TIE_TOLERANCE:
            better = True
        elif overlap < best_overlap - OVERLAP_TIE_TOLERANCE:
            better = False
        else:
            better = remainder > best_remainder

        if better:
            best_overlap = overlap
            best_remainder = remainder
            best_label = label

    if best_label is not None:
        return best_label

    if not turns:
        return UNKNOWN_SPEAKER_LABEL

    middle = (start + end) / 2
    nearest = min(
        turns,
        key=lambda turn: 0.0
        if turn[0] <= middle <= turn[1]
        else min(abs(middle - turn[0]), abs(middle - turn[1])),
    )
    return nearest[2]


def assign_speakers(chunks, turns):
    """각 전사 조각에 화자를 배정하여 (화자, 텍스트, 시작, 끝) 목록을 반환한다."""
    return [
        (pick_speaker(start, end, turns), text, start, end)
        for start, end, text in chunks
    ]


def smooth_assignments(assignments):
    """동일 화자 발화 사이에 아주 짧게 튄 반대쪽 화자 배정을 주변 화자로 흡수한다."""
    if SPEAKER_SMOOTH_MAX_CHUNKS <= 0 or len(assignments) < 3:
        return assignments, 0

    runs = []
    for index, (speaker, _, _, _) in enumerate(assignments):
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(index)
        else:
            runs.append([speaker, [index]])

    smoothed = list(assignments)
    absorbed = 0

    for position in range(1, len(runs) - 1):
        indices = runs[position][1]
        neighbor = runs[position - 1][0]
        if neighbor != runs[position + 1][0]:
            continue
        if len(indices) > SPEAKER_SMOOTH_MAX_CHUNKS:
            continue

        span = assignments[indices[-1]][3] - assignments[indices[0]][2]
        if span > SPEAKER_SMOOTH_MAX_SECONDS:
            continue

        if (
            assignments[indices[0]][2] - assignments[indices[0] - 1][3]
            > SPEAKER_SMOOTH_MAX_GAP
        ):
            continue
        if (
            assignments[indices[-1] + 1][2] - assignments[indices[-1]][3]
            > SPEAKER_SMOOTH_MAX_GAP
        ):
            continue

        for index in indices:
            _, text, start, end = smoothed[index]
            smoothed[index] = (neighbor, text, start, end)
        absorbed += 1

    return smoothed, absorbed


def join_chunk_text(previous, text):
    """앞선 텍스트와 현재 조각의 공백을 고려하여 이어 붙인다."""
    if text[:1].isspace():
        return previous + text
    return f"{previous} {text}"


def continues_line_by_timing(joined_text, start, previous_end):
    """무음 간격과 길이 상한만으로 이전 줄에 이어 붙일지 판정한다."""
    if previous_end is not None and start - previous_end > SPEAKER_MERGE_MAX_GAP:
        return False
    if 0 < SPEAKER_MERGE_MAX_CHARS < len(joined_text):
        return False
    return True


def continues_line(last_line, speaker, joined_text, start, previous_end):
    """현재 조각을 이전 줄에 병합할지 여부를 판정한다."""
    if last_line[0] != speaker:
        return False
    return continues_line_by_timing(joined_text, start, previous_end)


def merge_speaker_lines(assignments):
    """동일 화자의 연속된 발화 조각을 병합하여 '화자1: 문장' 목록을 생성한다."""
    merged = []
    previous_end = None

    for speaker, text, start, end in assignments:
        joined = join_chunk_text(merged[-1][1], text) if merged else None
        if merged and continues_line(
            merged[-1], speaker, joined, start, previous_end
        ):
            merged[-1][1] = joined
        else:
            merged.append([speaker, text.lstrip()])
        previous_end = end

    return [f"{speaker}: {text}" for speaker, text in merged]


def prepare_transcription_chunks(raw_chunks, audio_end=None):
    """Whisper 원시 조각을 절대 시간축으로 정규화해 (조각, 보정 횟수)를 반환한다."""
    chunks = normalize_chunks(raw_chunks, audio_end=audio_end)
    if not chunks:
        raise RuntimeError("타임스탬프가 있는 변환 조각이 없어 화자를 나눌 수 없습니다.")

    chunks, resets = repair_chunk_timeline(chunks)
    if resets:
        print(
            f"  타임스탬프가 {resets}번 뒤로 돌아가 이어 붙였습니다"
            " (창별 상대시간으로 나온 출력)."
        )
    return chunks, resets


def format_plain_text(chunks):
    """조각을 화자 표시 없이 읽기 좋은 줄로 합친다.

    줄은 화자가 아니라 무음 간격(SPEAKER_MERGE_MAX_GAP)과 길이 상한
    (SPEAKER_MERGE_MAX_CHARS)으로만 나눈다. 한 줄로 두면 익명화 기록의 line
    값이 전부 1이 된다.
    """
    lines = []
    previous_end = None

    for start, end, text in chunks:
        joined = join_chunk_text(lines[-1], text) if lines else None
        if lines and continues_line_by_timing(joined, start, previous_end):
            lines[-1] = joined
        else:
            lines.append(text.lstrip())
        previous_end = end

    print(f"  평문 정리: 조각 {len(chunks)}개 → {len(lines)}줄")
    return "\n".join(lines)


def format_prepared_diarized_text(chunks, turns, folded=None):
    """준비된 Whisper 조각과 화자 구간을 결합하여 '화자: 문장' 텍스트를 생성한다."""
    warn_timeline_mismatch(chunks, turns, folded)

    assignments, absorbed = smooth_assignments(assign_speakers(chunks, turns))
    if absorbed:
        print(f"  짧게 끼어든 배정 {absorbed}곳을 앞뒤 화자로 흡수했습니다.")

    lines = merge_speaker_lines(assignments)
    speakers = len({line.split(":", 1)[0] for line in lines})
    print(f"  화자 배정: 조각 {len(chunks)}개 → {len(lines)}줄, 화자 {speakers}명")

    return "\n".join(lines)


def transcribe_cropped_audio(
    audio_path,
    crop_start,
    crop_end,
    generate_kwargs,
    return_timestamps,
):
    """원본의 지정 구간을 PCM WAV로 잘라 Whisper에 다시 전달한다."""
    duration = max(0.0, crop_end - crop_start)
    if duration <= 0:
        raise RuntimeError("마지막 구간 재전사 범위가 비어 있습니다.")

    with tempfile.TemporaryDirectory(prefix="stt-tail-") as directory:
        tail_path = Path(directory) / "tail.wav"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ss",
            f"{crop_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(tail_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "마지막 구간 재전사에는 ffmpeg가 필요합니다."
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"종료 코드 {completed.returncode}"
            raise RuntimeError(f"마지막 음성 구간을 자르지 못했습니다: {detail}")

        with torch.inference_mode():
            result = transcriber(
                str(tail_path),
                batch_size=1,
                return_timestamps=return_timestamps,
                generate_kwargs=generate_kwargs,
            )
    return result, duration


def transcribe_tail_audio(audio_path, crop_start, crop_end, generate_kwargs):
    """마지막 구간을 단어 타임스탬프로 재전사해 원본 시간축으로 옮긴다."""
    result, duration = transcribe_cropped_audio(
        audio_path,
        crop_start,
        crop_end,
        generate_kwargs,
        "word",
    )
    raw_chunks = result.get("chunks") or []
    if not (result.get("text") or "").strip() or not raw_chunks:
        raise RuntimeError("마지막 구간 재전사에서도 텍스트를 얻지 못했습니다.")

    prepared = []
    for start, end, text in normalize_chunks(raw_chunks):
        # 30초 패딩 뒤의 결과는 crop 밖의 환각이므로 버린다.
        if start >= duration:
            continue
        end = min(end, duration)
        if end <= start:
            continue
        prepared.append((start + crop_start, end + crop_start, text))
    return prepared


def merge_time_intervals(intervals, merge_gap=0.0):
    """겹치거나 가까운 시간 구간을 합친다."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(interval) for interval in merged]


def speech_coverage_intervals(turns):
    """화자 구분은 버리고 pyannote가 검출한 전체 음성 합집합만 반환한다."""
    return merge_time_intervals(
        [(start, end) for start, end, _ in turns],
        merge_gap=0.10,
    )


def word_coverage_intervals(chunks):
    """Whisper 단어가 실제로 덮은 시간 구간을 작은 오차 여유와 함께 반환한다."""
    return merge_time_intervals(
        [
            (
                max(0.0, start - WORD_COVERAGE_PADDING_SECONDS),
                end + WORD_COVERAGE_PADDING_SECONDS,
            )
            for start, end, _ in chunks
        ],
        merge_gap=0.05,
    )


def find_missing_speech_intervals(turns, chunks):
    """pyannote 음성 중 Whisper 단어가 1초 이상 없는 모든 구간을 찾는다."""
    speech_intervals = speech_coverage_intervals(turns)
    covered = word_coverage_intervals(chunks)
    missing = []

    for speech_start, speech_end in speech_intervals:
        cursor = speech_start
        for word_start, word_end in covered:
            if word_end <= speech_start:
                continue
            if word_start >= speech_end:
                break
            if word_start > cursor:
                gap_end = min(word_start, speech_end)
                if gap_end - cursor >= MIN_MISSING_SPEECH_SECONDS:
                    missing.append((cursor, gap_end))
            cursor = max(cursor, min(word_end, speech_end))
            if cursor >= speech_end:
                break
        if speech_end - cursor >= MIN_MISSING_SPEECH_SECONDS:
            missing.append((cursor, speech_end))

    return merge_time_intervals(missing, merge_gap=0.10)


def split_missing_speech_intervals(intervals):
    """긴 누락 구간을 MISSING_SPEECH_MAX_RETRY_SECONDS 이하로 나눈다."""
    chunks = []
    for start, end in intervals:
        current = start
        while current < end:
            chunk_end = min(current + MISSING_SPEECH_MAX_RETRY_SECONDS, end)
            chunks.append((current, chunk_end))
            current = chunk_end
    return chunks


def merge_recovered_words(original, recovered):
    """같은 시각의 같은 단어는 버리고 복구 단어를 원래 시간축에 병합한다."""
    merged = list(original)
    for candidate in recovered:
        candidate_center = (candidate[0] + candidate[1]) / 2
        candidate_text = canonical_tail_text(candidate[2])
        duplicate = any(
            canonical_tail_text(text) == candidate_text
            and abs(((start + end) / 2) - candidate_center) <= 0.50
            for start, end, text in merged
        )
        if not duplicate:
            merged.append(candidate)
    return sorted(merged, key=lambda chunk: (chunk[0], chunk[1]))


def retry_missing_speech(audio_path, intervals, generate_kwargs, audio_seconds=None):
    """누락으로 판정된 음성 구간만 Whisper-large-v3로 다시 전사한다."""
    recovered = []
    retry_kwargs = dict(generate_kwargs)
    retry_kwargs["temperature"] = 0.0
    retry_kwargs["no_speech_threshold"] = NO_SPEECH_THRESHOLD

    for missing_start, missing_end in split_missing_speech_intervals(intervals):
        crop_start = max(0.0, missing_start - MISSING_SPEECH_RETRY_PADDING_SECONDS)
        crop_end = missing_end + MISSING_SPEECH_RETRY_PADDING_SECONDS
        if audio_seconds is not None:
            crop_end = min(crop_end, audio_seconds)
        print(
            f"  누락 음성 재전사: {missing_start:.2f}~{missing_end:.2f}초 "
            f"(입력 {crop_start:.2f}~{crop_end:.2f}초, "
            f"no_speech_threshold={NO_SPEECH_THRESHOLD:g})"
        )
        try:
            words = transcribe_tail_audio(
                audio_path, crop_start, crop_end, retry_kwargs
            )
        except (RuntimeError, ValueError) as exc:
            # 단어 토큰이 하나도 안 나온 crop에서만 나는 오류다. 복구할 단어가
            # 없다는 뜻이라 파일 전체를 실패시키지 않고, 다른 오류는 그대로 올린다.
            if "expected a non-empty list of Tensors" not in str(exc):
                raise
            print(
                "  누락 음성 재전사 결과 없음: "
                "Whisper가 단어 토큰을 생성하지 않았습니다."
            )
            continue

        recovered.extend(
            word
            for word in words
            if missing_start <= (word[0] + word[1]) / 2 <= missing_end
        )
    return recovered


def recover_missing_speech(audio_path, chunks, turns, generate_kwargs, audio_seconds=None):
    """오디오 전체에서 단어가 빠진 음성 구간을 찾아 제한적으로 복구한다."""
    combined = list(chunks)
    for retry_pass in range(1, MISSING_SPEECH_RETRY_LOOPS + 1):
        missing = find_missing_speech_intervals(turns, combined)
        if not missing:
            print(f"  누락 음성 검사 {retry_pass}: 공백 없음")
            break
        print(
            f"  누락 음성 검사 {retry_pass}/{MISSING_SPEECH_RETRY_LOOPS}: "
            f"{len(missing)}구간"
        )
        recovered = retry_missing_speech(
            audio_path, missing, generate_kwargs, audio_seconds
        )
        previous_count = len(combined)
        combined = merge_recovered_words(combined, recovered)
        added = len(combined) - previous_count
        print(f"  누락 음성 복구: 새 단어 {added}개 추가")
        if added <= 0:
            break
    return combined


def canonical_tail_text(text):
    """tail 중복 비교에서 띄어쓰기·문장부호 차이를 무시한 문자열이다."""
    return "".join(character.casefold() for character in text if character.isalnum())


def atomic_write_json(path, payload):
    """임시 파일을 거쳐 JSON을 원자적으로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def atomic_write_text(path, text):
    """임시 파일을 거쳐 텍스트 파일을 원자적으로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(text)
    os.replace(temporary_path, path)


def run_inference(
    audio_path,
    language=LANGUAGE,
    word_chunk_seconds=WORD_CHUNK_SECONDS,
    word_chunk_stride_seconds=WORD_CHUNK_STRIDE_SECONDS,
):
    """WAV를 전사하고 pyannote 음성 구간 대비 누락된 단어만 재전사한다.

    화자 ID는 쓰지 않으므로 반환 본문은 평문이다.
    """
    if transcriber is None or torch is None:
        raise RuntimeError("Whisper 모델이 아직 로드되지 않았습니다.")
    if diarizer is None:
        raise RuntimeError("누락 음성 검출용 pyannote 모델이 아직 로드되지 않았습니다.")

    # 누락 검출에 실제 단어 끝 시각이 필요하므로 항상 단어 단위로 받는다.
    return_timestamps = "word"
    chunk_kwargs = {}
    if word_chunk_seconds > 0:
        chunk_kwargs = {
            "chunk_length_s": word_chunk_seconds,
            "stride_length_s": (
                word_chunk_stride_seconds,
                word_chunk_stride_seconds,
            ),
        }
        print(
            f"  Whisper 외부 청크: {word_chunk_seconds:g}초, "
            f"앞뒤 겹침 {word_chunk_stride_seconds:g}초, 배치 1, 단어 타임스탬프"
        )
    else:
        print(
            "  Whisper native long-form: 외부 청크 없음, "
            "내부 복구용 타임스탬프 단어 단위"
        )

    audio_seconds = read_wav_duration(audio_path)
    if audio_seconds is None:
        print(
            "  주의: WAV 헤더와 ffprobe에서 길이를 읽지 못해 "
            "전체 길이 기준 검사를 건너뜁니다."
        )

    generate_kwargs = {
        "language": language,
        "task": "transcribe",
        "temperature": TEMPERATURE_FALLBACK,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
        "logprob_threshold": LOGPROB_THRESHOLD,
        "no_speech_threshold": NO_SPEECH_THRESHOLD,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "condition_on_prev_tokens": False,
    }

    started_at = time.monotonic()
    with torch.inference_mode():
        result = transcriber(
            str(audio_path),
            batch_size=1,
            return_timestamps=return_timestamps,
            **chunk_kwargs,
            generate_kwargs=generate_kwargs,
        )

    primary_text = (result.get("text") or "").strip()
    if not primary_text:
        raise RuntimeError("변환 결과가 비어 있습니다.")
    chunks = normalize_chunks(result.get("chunks"), audio_end=audio_seconds)
    if not chunks:
        raise RuntimeError(
            "타임스탬프 조각이 없어 별도 화자분리에 전달할 수 없습니다."
        )

    elapsed = time.monotonic() - started_at
    print(f"  변환 완료: {elapsed:.1f}초")

    chunks, folded = repair_chunk_timeline(chunks)
    if folded:
        print(f"  타임스탬프가 {folded}번 뒤로 돌아가 이어 붙였습니다.")

    # 화자 라벨은 버리고 음성 구간의 합집합만 누락 판정에 쓴다.
    turns = run_diarization(
        audio_path,
        num_speakers=DEFAULT_NUM_SPEAKERS,
        min_speakers=DEFAULT_MIN_SPEAKERS,
    )
    chunks = recover_missing_speech(
        audio_path,
        chunks,
        turns,
        generate_kwargs,
        audio_seconds,
    )
    text = format_plain_text(chunks)

    transcript_end, coverage = transcript_coverage(chunks, audio_seconds)
    short_transcript = warn_short_transcript(
        transcript_end, audio_seconds, coverage, folded
    )
    if coverage is not None and not short_transcript:
        print(f"  전사 범위: {transcript_end:.1f}/{audio_seconds:.1f}초 ({coverage:.0%})")

    clear_gpu_cache()
    return {
        "text": text,
        "timestamp_unit": "word",
        "chunks": chunks,
        "audio_seconds": audio_seconds,
        "transcript_end": transcript_end,
        "coverage": coverage,
        "short_transcript": short_transcript,
    }


def find_wav_files(input_dir):
    """입력 폴더 아래의 WAV 파일을 경로순으로 반환한다."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def limit_sources(sources, sample, unit):
    """샘플 실행용으로 경로순 앞에서 sample개만 남긴다(재현성을 위해 무작위 아님)."""
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
    길이가 비슷한 녹음이 몰려 있어 블록으로 자르면 부하가 한쪽에 쏠린다.
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
    timestamps_folder=STT_TIMESTAMPS_FOLDER,
    sample=None,
    shard=None,
):
    """(입력 WAV, 평문 TXT, 타임스탬프 JSON) 경로 묶음을 생성한다."""
    input_dir = Path(input_folder).expanduser().resolve()
    output_dir = Path(output_folder).expanduser().resolve()
    timestamps_dir = Path(timestamps_folder).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"입력 경로는 폴더여야 합니다: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"출력 경로는 폴더여야 합니다: {output_dir}")
    if timestamps_dir.exists() and not timestamps_dir.is_dir():
        raise ValueError(f"타임스탬프 경로는 폴더여야 합니다: {timestamps_dir}")
    if timestamps_dir == output_dir:
        raise ValueError(
            "평문과 타임스탬프 폴더가 같으면 산출물이 섞입니다: "
            f"{output_dir}"
        )

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        raise ValueError(f"입력 폴더에 WAV 파일이 없습니다: {input_dir}")

    # 샘플이 먼저다. 순서를 바꾸면 샤드마다 --sample개씩 처리하게 된다.
    wav_files = limit_sources(wav_files, sample, "WAV 파일")
    wav_files = shard_sources(wav_files, shard, "WAV 파일")

    return [
        (
            source,
            output_dir / source.relative_to(input_dir).with_suffix(".txt"),
            timestamps_dir / source.relative_to(input_dir).with_suffix(".json"),
        )
        for source in wav_files
    ]


def legacy_stt_state_path(output_path):
    """구버전 TXT 인접 상태 파일 경로를 반환한다."""
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".stt.json")


def stt_state_path(
    output_path,
    output_folder=OUTPUT_FOLDER,
    cache_folder=STT_CACHE_FOLDER,
):
    """STT-cache 폴더 내의 상태 파일 경로를 생성한다."""
    output_path = Path(output_path).expanduser().resolve()
    output_dir = Path(output_folder).expanduser().resolve()
    cache_dir = Path(cache_folder).expanduser().resolve()
    try:
        relative = output_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(
            f"STT 출력 파일이 출력 폴더 밖에 있습니다: {output_path}"
        ) from exc
    return (cache_dir / relative).with_suffix(relative.suffix + ".stt.json")


def migrate_legacy_stt_state(output_path, state_path):
    """TXT 인접 상태 파일을 STT-cache 폴더로 마이그레이션한다."""
    legacy_path = legacy_stt_state_path(output_path)
    state_path = Path(state_path)
    if not legacy_path.is_file():
        return False
    if state_path.exists():
        legacy_path.unlink()
        print(f"  TXT 폴더의 중복 STT 상태 파일 제거: {legacy_path}")
        return True
    state_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(legacy_path, state_path)
    print(f"  기존 STT 상태 파일 이동: {state_path}")
    return True


def build_stt_fingerprint(
    source_path,
    language,
    word_chunk_seconds=WORD_CHUNK_SECONDS,
    word_chunk_stride_seconds=WORD_CHUNK_STRIDE_SECONDS,
):
    """재개(resume) 검증을 위한 STT 설정 및 원본 메타데이터 지문을 생성한다."""
    source_stat = Path(source_path).stat()
    return {
        "source": {
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "model_path": str(Path(MODEL_PATH).expanduser()),
        "language": language,
        "missing_speech_recovery": {
            "detector": "pyannote",
            "min_gap_seconds": MIN_MISSING_SPEECH_SECONDS,
            "retry_padding_seconds": MISSING_SPEECH_RETRY_PADDING_SECONDS,
            "max_retry_chunk_seconds": MISSING_SPEECH_MAX_RETRY_SECONDS,
            "word_coverage_padding_seconds": WORD_COVERAGE_PADDING_SECONDS,
            "retry_loops": MISSING_SPEECH_RETRY_LOOPS,
        },
        "native_long_form": not (word_chunk_seconds > 0),
        "external_chunk_length_seconds": (
            word_chunk_seconds if word_chunk_seconds > 0 else None
        ),
        "external_chunk_stride_seconds": (
            word_chunk_stride_seconds if word_chunk_seconds > 0 else None
        ),
        "decoding": {
            "temperature_fallback": list(TEMPERATURE_FALLBACK),
            "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
            "logprob_threshold": LOGPROB_THRESHOLD,
            "no_speech_threshold": NO_SPEECH_THRESHOLD,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "condition_on_prev_tokens": False,
        },
    }


def read_stt_state(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def mark_stt_complete(path, fingerprint, coverage=None):
    """STT 완료 상태를 캐시 파일에 저장한다. coverage를 주면 함께 남긴다."""
    state = read_stt_state(path)
    if not isinstance(state, dict) or state.get("fingerprint") != fingerprint:
        state = {
            "version": 1,
            "fingerprint": fingerprint,
        }
    state["status"] = "complete"
    if coverage is not None:
        state["coverage"] = coverage
    atomic_write_json(path, state)


def recorded_short_transcript(state_path):
    """캐시 상태 파일의 뒷부분 누락 판정을 (미달 여부, 비율)로 읽는다.

    coverage 블록이 없는 옛 상태 파일은 판정 없음으로 본다.
    """
    state = read_stt_state(state_path)
    if not isinstance(state, dict):
        return False, None
    coverage = state.get("coverage")
    if not isinstance(coverage, dict) or not coverage.get("short_transcript"):
        return False, None

    ratio = coverage.get("ratio")
    return True, ratio if isinstance(ratio, (int, float)) else None


def write_run_summary(path, summary):
    """run_pipeline.py가 읽을 단계 집계를 JSON으로 저장한다(경로·사유는 제외)."""
    document = {
        "total": summary["total"],
        "succeeded": summary["succeeded"],
        "skipped": summary["skipped"],
        "failed": summary["failed"],
    }
    # 1단계만 세는 값이라 없을 수도 있다. 파일 경로는 담지 않고 개수만 쓴다.
    if "short_transcripts" in summary:
        document["short_transcripts"] = len(summary["short_transcripts"])
    atomic_write_json(path, document)


def build_timestamp_document(transcription, file_label):
    """화자분리 단계가 읽을 타임스탬프 JSON 문서를 만든다.

    version을 올리면 기존 JSON 전부가 read_timestamp_document 검증에서 떨어진다.
    """
    return {
        "version": 1,
        "file": file_label,
        "unit": transcription["timestamp_unit"],
        "coverage": {
            "audio_seconds": transcription.get("audio_seconds"),
            "transcript_end": transcription.get("transcript_end"),
            "ratio": transcription.get("coverage"),
            "min_ratio": TIMELINE_COVERAGE_MIN_RATIO,
            "short_transcript": bool(transcription.get("short_transcript")),
        },
        "chunks": [
            {"start": start, "end": end, "text": text}
            for start, end, text in transcription["chunks"]
        ],
    }


def read_timestamp_document(path):
    """타임스탬프 JSON을 검증하고 Whisper 원시 조각 모양으로 반환한다."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"타임스탬프 JSON을 읽지 못했습니다: {path}") from exc

    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError(f"지원하지 않는 타임스탬프 JSON입니다: {path}")
    if document.get("unit") not in ("word", "segment"):
        raise ValueError(f"타임스탬프 단위가 올바르지 않습니다: {path}")
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"타임스탬프 조각이 비어 있습니다: {path}")

    raw_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"타임스탬프 조각 {index}이 객체가 아닙니다: {path}")
        start = chunk.get("start")
        end = chunk.get("end")
        text = chunk.get("text")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"타임스탬프 조각 {index}의 시간이 잘못됐습니다: {path}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"타임스탬프 조각 {index}의 본문이 비어 있습니다: {path}")
        raw_chunks.append(
            {"timestamp": (float(start), float(end)), "text": text}
        )

    return {**document, "chunks": raw_chunks}


def report_recorded_coverage(document):
    """타임스탬프 JSON에 기록된 뒷부분 누락 판정을 화자분리 로그에도 다시 알린다."""
    coverage = document.get("coverage")
    if not isinstance(coverage, dict) or not coverage.get("short_transcript"):
        return False

    ratio = coverage.get("ratio")
    shown = f"{ratio:.0%}" if isinstance(ratio, (int, float)) else "측정 불가"
    print(
        f"  주의: STT 단계에서 뒷부분 누락으로 판정된 파일입니다(커버리지 {shown}). "
        "화자분리가 없는 본문을 만들어 낼 수는 없습니다."
    )
    return True


def reusable_timestamp_output(source_path, timestamp_path):
    """--resume에서 재사용할 수 있는 타임스탬프 JSON인지 확인한다."""
    source_path = Path(source_path)
    timestamp_path = Path(timestamp_path)
    if timestamp_path.name != source_path.with_suffix(".json").name:
        return False
    try:
        read_timestamp_document(timestamp_path)
        return True
    except ValueError:
        return False


def reusable_text_output(
    source_path,
    output_path,
):
    """--resume 옵션 시 재사용 가능한 기존 TXT 결과가 있는지 확인한다."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    expected_name = source_path.with_suffix(".txt").name
    try:
        if output_path.name != expected_name:
            return False
        if not output_path.is_file():
            return False
        return bool(output_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def process_folder(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    timestamps_folder=STT_TIMESTAMPS_FOLDER,
    language=LANGUAGE,
    word_chunk_seconds=WORD_CHUNK_SECONDS,
    word_chunk_stride_seconds=WORD_CHUNK_STRIDE_SECONDS,
    sample=None,
    shard=None,
    resume=False,
    stt_cache_folder=STT_CACHE_FOLDER,
    ensure_model_loaded=None,
):
    """WAV 폴더를 순회하며 평문 TXT와 타임스탬프 JSON을 저장한다."""
    cache_dir = Path(stt_cache_folder).expanduser().resolve()
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError(f"STT 캐시 경로는 폴더여야 합니다: {cache_dir}")

    jobs = build_jobs(input_folder, output_folder, timestamps_folder, sample, shard)
    input_dir = Path(input_folder).expanduser().resolve()
    total = len(jobs)
    succeeded = 0
    skipped = 0
    failures = []
    short_transcripts = []

    for index, (source_path, output_path, timestamp_path) in enumerate(
        jobs, start=1
    ):
        state_path = stt_state_path(
            output_path,
            output_folder=output_folder,
            cache_folder=stt_cache_folder,
        )
        migrate_legacy_stt_state(output_path, state_path)

        # 평문과 타임스탬프가 모두 있어야 화자분리까지 재현할 수 있다.
        if (
            resume
            and reusable_text_output(source_path, output_path)
            and reusable_timestamp_output(source_path, timestamp_path)
        ):
            skipped += 1
            print(f"[{index}/{total}] 기존 STT 결과 건너뜀: {output_path}")
            # --resume은 TXT가 비었는지만 보므로 건너뛴 파일의 판정도 다시 넣는다.
            was_short, recorded_ratio = recorded_short_transcript(state_path)
            if was_short:
                short_transcripts.append((source_path, recorded_ratio))
                print(
                    "  주의: 예전 실행에서 뒷부분 누락으로 판정된 파일입니다. "
                    "다시 전사하려면 이 TXT를 지우거나 --resume 없이 실행하세요."
                )
            continue

        fingerprint = build_stt_fingerprint(
            source_path,
            language,
            word_chunk_seconds,
            word_chunk_stride_seconds,
        )

        if not resume:
            atomic_write_json(
                state_path,
                {
                    "version": 1,
                    "fingerprint": fingerprint,
                    "status": "in_progress",
                },
            )

        if ensure_model_loaded is not None:
            ensure_model_loaded()

        print(f"[{index}/{total}] 변환 중: {source_path}")
        try:
            transcription = run_inference(
                audio_path=source_path,
                language=language,
                word_chunk_seconds=word_chunk_seconds,
                word_chunk_stride_seconds=word_chunk_stride_seconds,
            )
            atomic_write_text(output_path, transcription["text"] + "\n")
            print(f"  저장 완료: {output_path}")
            atomic_write_json(
                timestamp_path,
                build_timestamp_document(
                    transcription,
                    source_path.relative_to(input_dir).as_posix(),
                ),
            )
            print(f"  타임스탬프 저장: {timestamp_path}")
            coverage = {
                "audio_seconds": transcription.get("audio_seconds"),
                "transcript_end": transcription.get("transcript_end"),
                "ratio": transcription.get("coverage"),
                "min_ratio": TIMELINE_COVERAGE_MIN_RATIO,
                "short_transcript": bool(transcription.get("short_transcript")),
            }
            mark_stt_complete(state_path, fingerprint, coverage)
            if coverage["short_transcript"]:
                short_transcripts.append((source_path, coverage["ratio"]))
            succeeded += 1
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((source_path, str(exc)))
            print(f"  실패: {exc}")
        finally:
            clear_gpu_cache()

    return {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
        # 실패가 아니라, 저장은 됐지만 뒷부분이 비어 있을 수 있는 파일이다.
        "short_transcripts": short_transcripts,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="로컬 Whisper 모델로 WAV 폴더를 TXT로 일괄 변환"
    )
    parser.add_argument(
        "--input-folder",
        default=INPUT_FOLDER,
        help="입력 WAV 폴더 (기본값: 실행 파일 폴더/Original)",
    )
    parser.add_argument(
        "--output-folder",
        default=OUTPUT_FOLDER,
        help="평문 전사 TXT 폴더 (기본값: 실행 파일 폴더/Text)",
    )
    parser.add_argument(
        "--timestamps-folder",
        default=STT_TIMESTAMPS_FOLDER,
        help=(
            "별도 화자분리용 단어·구간 타임스탬프 JSON 폴더 "
            "(기본값: 실행 파일 폴더/STT-Timestamps)"
        ),
    )
    parser.add_argument(
        "--stt-cache-folder",
        default=STT_CACHE_FOLDER,
        help=(
            "STT 재개 상태 JSON 폴더 "
            "(기본값: 실행 파일 폴더/STT-cache, text 하위 구조 유지)"
        ),
    )
    parser.add_argument(
        "--language",
        default=LANGUAGE,
        help="음성 언어: korean, english, japanese 등",
    )
    parser.add_argument(
        "--diarization-model",
        default=DIARIZATION_MODEL_PATH,
        help=(
            "누락 음성 검출용 pyannote 파이프라인 폴더/config.yaml 경로 또는 "
            "허브 ID (기본값: %(default)s)"
        ),
    )
    parser.add_argument(
        "--word-chunk-seconds",
        type=float,
        default=WORD_CHUNK_SECONDS,
        help=(
            "단어 타임스탬프 외부 청크 길이(초, 기본값: %(default)s, "
            "0이면 native long-form)"
        ),
    )
    parser.add_argument(
        "--word-chunk-stride-seconds",
        type=float,
        default=WORD_CHUNK_STRIDE_SECONDS,
        help="외부 청크 앞뒤 겹침 길이(초, 기본값: %(default)s)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help=(
            "샘플 실행: 경로순 앞에서 이 개수만큼만 변환 "
            "(예: --sample 10, 기본값은 전체 변환)"
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
            "평문 TXT와 타임스탬프 JSON이 둘 다 정상일 때 STT를 건너뜀 "
            "(WAV 수정 시각·내용·STT 설정 변경은 비교하지 않음)"
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
    if args.word_chunk_seconds < 0:
        parser.error("--word-chunk-seconds는 0 이상이어야 합니다")
    if args.word_chunk_stride_seconds < 0:
        parser.error("--word-chunk-stride-seconds는 0 이상이어야 합니다")
    if (
        args.word_chunk_seconds > 0
        and args.word_chunk_stride_seconds * 2 >= args.word_chunk_seconds
    ):
        parser.error("청크 앞뒤 겹침의 합은 청크 길이보다 작아야 합니다")
    if args.shard is not None:
        try:
            args.shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)

    model_loaded = False

    def ensure_model_loaded():
        nonlocal model_loaded
        if model_loaded:
            return
        load_model(whisper_model_path=MODEL_PATH)
        load_diarizer(model_path=args.diarization_model)
        model_loaded = True

    try:
        summary = process_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            timestamps_folder=args.timestamps_folder,
            language=args.language,
            word_chunk_seconds=args.word_chunk_seconds,
            word_chunk_stride_seconds=args.word_chunk_stride_seconds,
            sample=args.sample,
            shard=args.shard,
            resume=args.resume,
            stt_cache_folder=args.stt_cache_folder,
            ensure_model_loaded=ensure_model_loaded,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        return 2

    if args.summary_json:
        try:
            write_run_summary(args.summary_json, summary)
        except OSError as exc:
            # 집계 저장 실패로 정상 처리된 결과를 실패로 만들지는 않는다.
            print(f"경고: 처리 결과 요약을 저장하지 못했습니다: {exc}")
    short_transcripts = summary.get("short_transcripts") or []

    print("\n처리 결과:")
    print(f"  전체: {summary['total']}개")
    print(f"  성공: {summary['succeeded']}개")
    print(f"  건너뜀: {summary['skipped']}개")
    print(f"  실패: {summary['failed']}개")
    print(f"  뒷부분 누락 의심: {len(short_transcripts)}개")
    print(f"  평문 전사: {args.output_folder}")
    print(f"  타임스탬프: {args.timestamps_folder}")

    if short_transcripts:
        # 실패가 아니므로 종료 코드는 바꾸지 않고 목록만 다시 보여 준다.
        print(
            f"\n전사가 오디오 길이의 {TIMELINE_COVERAGE_MIN_RATIO:.0%}에 못 미친 파일:"
        )
        for source_path, ratio in short_transcripts:
            shown = f"{ratio:.0%}" if ratio is not None else "측정 불가"
            print(f"  - {source_path}: 커버리지 {shown}")
        print(
            "  이 파일들의 TXT는 저장됐지만 뒷부분이 비어 있을 수 있습니다. "
            "STT-Timestamps/의 coverage 블록과 함께 확인하세요."
        )

    if summary["failures"]:
        print("\n실패한 파일:")
        for source_path, reason in summary["failures"]:
            print(f"  - {source_path}: {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
