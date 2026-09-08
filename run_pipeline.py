"""STT, 화자분리, 개인정보 익명화를 독립 프로세스로 연속 실행한다.

GPU 메모리 격리를 위해 각 단계를 별도 subprocess로 돌린다. --gpus는 카드마다
프로세스를 띄우고, --shard가 경로순으로 파일을 나눈다(CUDA_VISIBLE_DEVICES 배정).

파이프라인 흐름:
    1. Original/ ──stt_server.py──► Text/ + STT-Timestamps/
    2. Original/ + STT-Timestamps/ ──diarization_server.py──► Diarization/
    3. Text/ ──anonimizer_server.py──► Anonymization/ + Anonymization_info/
    4. Diarization/ ──anonimizer_server.py──► Anonymization_diar/
       (--input-kind diarized --no-info: 치환 값은 3단계 기록에 이미 있다)

사용 예시:
    python run_pipeline.py                        # 기본 폴더로 4단계 전체 실행
    python run_pipeline.py --sample 10            # 상위 10개만 샘플 실행
    python run_pipeline.py --resume               # 단계별 기존 결과 재사용
    python run_pipeline.py --skip-stt             # 기존 타임스탬프로 2단계부터
    python run_pipeline.py -- --language korean   # -- 뒤는 STT 단계에 직접 전달
    python run_pipeline.py --gpus auto            # nvidia-smi가 보는 카드 전부
    python run_pipeline.py --workers-per-gpu 4    # 카드 한 장에서 STT만 4중 실행
    python run_pipeline.py --gpus 1,2 --quiet-shards   # 자식 출력은 로그 파일에만
"""

import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent

STT_SCRIPT = SERVER_DIR / "stt_server.py"
DIARIZATION_SCRIPT = SERVER_DIR / "diarization_server.py"
ANONYMIZER_SCRIPT = SERVER_DIR / "anonimizer_server.py"

WAV_FOLDER = os.environ.get(
    "WAV_FOLDER",
    os.environ.get("INPUT_FOLDER", str(SERVER_DIR / "Original")),
)
TEXT_FOLDER = os.environ.get("TEXT_FOLDER", str(SERVER_DIR / "Text"))
DIARIZATION_FOLDER = os.environ.get(
    "DIARIZATION_FOLDER", str(SERVER_DIR / "Diarization")
)
STT_CACHE_FOLDER = os.environ.get(
    "STT_CACHE_FOLDER", str(SERVER_DIR / "STT-cache")
)
STT_TIMESTAMPS_FOLDER = os.environ.get(
    "STT_TIMESTAMPS_FOLDER", str(SERVER_DIR / "STT-Timestamps")
)
RESULT_FOLDER = os.environ.get(
    "RESULT_FOLDER",
    os.environ.get("OUTPUT_FOLDER", str(SERVER_DIR / "Anonymization")),
)
INFO_FOLDER = os.environ.get("INFO_FOLDER", str(SERVER_DIR / "Anonymization_info"))
DIARIZATION_RESULT_FOLDER = os.environ.get(
    "DIARIZATION_RESULT_FOLDER", str(SERVER_DIR / "Anonymization_diar")
)

# 자식 프로세스에 상속하지 않을 폴더 환경 변수 목록
SHARED_FOLDER_VARS = (
    "INPUT_FOLDER",
    "OUTPUT_FOLDER",
    "INFO_FOLDER",
    "STT_CACHE_FOLDER",
    "STT_TIMESTAMPS_FOLDER",
    "DIARIZATION_FOLDER",
)

EXIT_INTERRUPTED = 130

LOG_FOLDER = os.environ.get("LOG_FOLDER", str(SERVER_DIR / "logs"))
SHARD_POLL_SECONDS = 5.0
SHARD_READER_JOIN_SECONDS = 10.0

# nvidia-smi -L의 "GPU 0: NVIDIA ..." 줄에서 인덱스만 뽑는다.
NVIDIA_SMI_GPU_PATTERN = re.compile(r"^\s*GPU\s+(\d+)\s*:")


def build_child_env():
    """자식 프로세스에 전달할 환경 변수를 생성한다 (폴더 충돌 변수 제거)."""
    env = os.environ.copy()
    for name in SHARED_FOLDER_VARS:
        env.pop(name, None)
    return env


def read_run_summary(path):
    """자식이 남긴 집계 JSON을 읽는다. 없거나 깨졌으면 None을 돌려준다."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if not all(
        isinstance(document.get(key), int)
        for key in ("total", "succeeded", "skipped", "failed")
    ):
        return None
    # 1단계만 남기는 선택 항목이다. 모양이 다르면 이 값만 버린다.
    if "short_transcripts" in document and not isinstance(
        document["short_transcripts"], int
    ):
        del document["short_transcripts"]
    return document


def detect_gpus():
    """nvidia-smi -L이 보여 주는 GPU 인덱스 목록을 문자열로 반환한다."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return [
        match.group(1)
        for match in (
            NVIDIA_SMI_GPU_PATTERN.match(line)
            for line in completed.stdout.splitlines()
        )
        if match
    ]


def resolve_gpus(spec):
    """--gpus 값을 GPU 인덱스 목록으로 바꾼다. 지정이 없으면 None이다."""
    if spec is None:
        return None

    if spec.strip().lower() == "auto":
        detected = detect_gpus()
        if not detected:
            raise ValueError(
                "nvidia-smi로 GPU를 찾지 못했습니다. "
                "--gpus에 인덱스를 직접 지정하세요 (예: --gpus 0,1,2,3)."
            )
        print(f"GPU 자동 감지: {', '.join(detected)} ({len(detected)}장)")
        return detected

    devices = [piece.strip() for piece in spec.split(",") if piece.strip()]
    if not devices:
        raise ValueError(f"--gpus 값이 비어 있습니다: {spec}")
    for device in devices:
        if not device.isdigit():
            raise ValueError(
                f"--gpus는 쉼표로 구분한 GPU 인덱스여야 합니다: {spec}"
            )
    if len(set(devices)) != len(devices):
        raise ValueError(f"--gpus에 같은 인덱스가 중복됐습니다: {spec}")
    return devices


def build_shard_plan(gpus, workers_per_gpu):
    """같은 GPU의 워커를 연속 배치한 (샤드 번호, GPU) 목록을 만든다."""
    devices = list(gpus) if gpus else [None]
    return list(
        enumerate(
            device for device in devices for _ in range(workers_per_gpu)
        )
    )


def shard_labels(plan):
    """GPU·워커 번호로 너비를 맞춘 출력 라벨을 만든다. GPU 미지정 시 샤드 번호를 쓴다."""
    per_device = {}
    for _, device in plan:
        per_device[device] = per_device.get(device, 0) + 1

    used = {}
    labels = []
    for shard_index, device in plan:
        if device is None:
            labels.append(f"샤드 {shard_index}")
            continue
        if per_device[device] == 1:
            labels.append(f"GPU {device}")
            continue
        ordinal = used.get(device, 0)
        used[device] = ordinal + 1
        labels.append(f"GPU {device}.{ordinal}")

    width = max(len(label) for label in labels)
    return [label.ljust(width) for label in labels]


def collapse_progress_line(line):
    """\\r로 덮어쓰는 진행 표시에서 마지막 상태만 남긴다."""
    return line.rsplit("\r", 1)[-1]


def stream_child_output(process, label, handle, lock):
    """자식 출력을 이름표를 붙여 터미널로 흘리면서 로그 파일에도 그대로 쓴다.

    os.read는 텍스트 스트림의 read(n)과 달리 버퍼가 차기를 기다리지 않아 진행
    상황이 바로 보인다. 로그 파일에는 접지 않은 원본을 쓴다.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stream = process.stdout
    pending = ""
    while True:
        try:
            data = os.read(stream.fileno(), 8192)
        except OSError:
            break
        if not data:
            break
        text = decoder.decode(data)
        if not text:
            continue
        write_log(handle, text)
        pending += text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            with lock:
                print(f"[{label}] {collapse_progress_line(line)}", flush=True)
    # 줄바꿈 없이 끝난 마지막 조각도 버리지 않는다.
    if pending.strip():
        with lock:
            print(f"[{label}] {collapse_progress_line(pending)}", flush=True)
    write_log(handle, "")


def write_log(handle, text):
    """로그를 기록하고 flush한다. 부모가 먼저 닫은 핸들의 오류는 무시한다."""
    try:
        if text:
            handle.write(text)
        handle.flush()
    except (ValueError, OSError):
        pass


def worst_exit_code(codes):
    """샤드 종료 코드 중 가장 나쁜 것을 고른다.

    심각도는 0 < 1 < 2지만 신호로 죽은 자식은 음수라 max()로 잡히지 않으므로,
    예상 밖의 코드를 먼저 돌려준다.
    """
    unexpected = [code for code in codes if code not in (0, 1, 2)]
    if unexpected:
        return unexpected[0]
    return max(codes, default=0)


def aggregate_summaries(summaries):
    """샤드별 집계를 합쳐 (합계, 읽지 못한 샤드 수)를 반환한다."""
    readable = [summary for summary in summaries if summary is not None]
    missing = len(summaries) - len(readable)
    if not readable:
        return None, missing
    combined = {
        key: sum(summary[key] for summary in readable)
        for key in ("total", "succeeded", "skipped", "failed")
    }
    # 1단계만 남기는 값이라 있을 때만 더한다.
    if any("short_transcripts" in summary for summary in readable):
        combined["short_transcripts"] = sum(
            summary.get("short_transcripts", 0) for summary in readable
        )
    return combined, missing


def run_stage(
    title,
    script,
    arguments,
    summary_dir=None,
    gpus=None,
    workers_per_gpu=1,
    log_folder=None,
    summary_name=None,
    stream=True,
):
    """단계 스크립트를 실행하고 (종료 코드, 집계, 집계 유실 샤드 수)를 반환한다.

    집계는 자식이 --summary-json으로 남긴 파일에서 읽는다(로그 파싱보다 안전).
    샤드가 하나면 자식이 터미널에 그대로 출력하고 로그 파일은 없다.

    summary_name은 집계·로그 파일 이름이다. 같은 스크립트를 두 번 돌리는 3·4단계는
    이름이 달라야 뒤 단계가 앞 단계의 집계를 덮어쓰지 않는다.
    """
    plan = build_shard_plan(gpus, workers_per_gpu)
    stem = summary_name or Path(script).stem
    if len(plan) == 1:
        return run_single_stage(
            title, script, arguments, summary_dir, plan[0][1], stem
        )
    return run_sharded_stage(
        title,
        script,
        arguments,
        summary_dir,
        plan,
        Path(log_folder or LOG_FOLDER).expanduser(),
        stem,
        stream,
    )


def run_single_stage(
    title,
    script,
    arguments,
    summary_dir=None,
    device=None,
    summary_name=None,
):
    """단계를 단일 프로세스로 실행한다 (자식 출력은 터미널로 그대로)."""
    stem = summary_name or Path(script).stem
    summary_path = None
    if summary_dir is not None:
        summary_path = Path(summary_dir) / f"{stem}.json"
        arguments = [*arguments, "--summary-json", str(summary_path)]

    env = build_child_env()
    if device is not None:
        # 나누지는 않지만 카드는 고른다. 자식 안에서 이 카드가 cuda:0이다.
        env["CUDA_VISIBLE_DEVICES"] = device

    command = [sys.executable, str(script), *arguments]
    print(f"\n=== {title} ===")
    if device is not None:
        print(f"GPU {device}번 사용")
    print(f"실행: {' '.join(command)}", flush=True)

    started_at = time.monotonic()
    completed = subprocess.run(command, cwd=str(SERVER_DIR), env=env)
    elapsed = time.monotonic() - started_at

    print(f"=== {title} 종료: 코드 {completed.returncode}, {elapsed:.1f}초 ===")
    summary = read_run_summary(summary_path) if summary_path else None
    return completed.returncode, summary, 0


def run_sharded_stage(
    title,
    script,
    arguments,
    summary_dir,
    plan,
    log_dir,
    summary_name=None,
    stream=True,
):
    """단계를 샤드별 자식 프로세스로 동시에 실행한다.

    stream이 True면 [GPU 2] 같은 이름표를 붙여 터미널과 로그 파일에 함께 쓴다.
    False면(--quiet-shards) 자식 stdout을 로그 파일에 직접 연결한다.
    """
    stem = summary_name or Path(script).stem
    total_shards = len(plan)
    labels = shard_labels(plan)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {title} ===")
    print(f"{total_shards}개 샤드로 나누어 동시에 실행합니다.")

    processes = []
    handles = []
    readers = []
    print_lock = threading.Lock()
    summary_paths = []
    started_at = time.monotonic()

    try:
        for shard_index, device in plan:
            env = build_child_env()
            if device is not None:
                # 자식 안에서 이 카드가 cuda:0이 되므로 스크립트를 고칠 필요가 없다.
                env["CUDA_VISIBLE_DEVICES"] = device
            if stream:
                # 파이프로 받으면 파이썬이 블록 버퍼링을 써서 출력이 뭉텅이로 온다.
                env["PYTHONUNBUFFERED"] = "1"

            command = [
                sys.executable,
                str(script),
                *arguments,
                "--shard",
                f"{shard_index}/{total_shards}",
            ]
            summary_path = None
            if summary_dir is not None:
                summary_path = (
                    Path(summary_dir) / f"{stem}.shard{shard_index}.json"
                )
                command += ["--summary-json", str(summary_path)]
            summary_paths.append(summary_path)

            log_path = log_dir / f"{stem}.shard{shard_index}.log"
            handle = open(log_path, "w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                command,
                cwd=str(SERVER_DIR),
                env=env,
                stdout=subprocess.PIPE if stream else handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            if stream:
                reader = threading.Thread(
                    target=stream_child_output,
                    args=(process, labels[shard_index], handle, print_lock),
                    daemon=True,
                )
                reader.start()
                readers.append(reader)
            where = f"GPU {device}" if device is not None else "기본 GPU"
            print(f"  샤드 {shard_index}/{total_shards}  {where}  로그: {log_path}")

        if stream:
            print("  자식 출력은 [이름표]를 붙여 아래에 그대로 흘립니다.", flush=True)
        else:
            print(
                f"  진행 상황: tail -f {log_dir / f'{stem}.shard*.log'}",
                flush=True,
            )

        codes = [None] * total_shards
        remaining = list(range(total_shards))
        while remaining:
            time.sleep(SHARD_POLL_SECONDS)
            for shard_index in list(remaining):
                code = processes[shard_index].poll()
                if code is None:
                    continue
                codes[shard_index] = code
                remaining.remove(shard_index)
                with print_lock:
                    print(
                        f"  [{labels[shard_index]}] 종료: 코드 {code}, "
                        f"{time.monotonic() - started_at:.1f}초 "
                        f"(남은 샤드 {len(remaining)}개)",
                        flush=True,
                    )
    except BaseException:
        # Ctrl+C 포함. 자식을 남기면 GPU 메모리를 붙든 고아가 된다.
        terminate_processes(processes)
        raise
    finally:
        # 읽기 스레드를 먼저 기다린다. 로그 핸들을 먼저 닫으면 마지막 줄이 사라진다.
        for reader in readers:
            reader.join(timeout=SHARD_READER_JOIN_SECONDS)
        for process in processes:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
        for handle in handles:
            handle.close()

    elapsed = time.monotonic() - started_at
    worst = worst_exit_code(codes)
    summary, missing = aggregate_summaries(
        [read_run_summary(path) if path else None for path in summary_paths]
    )
    print(f"=== {title} 종료: 코드 {worst}, {elapsed:.1f}초 ===")
    return worst, summary, missing


def terminate_processes(processes):
    """살아 있는 자식에게 종료를 요청하고, 듣지 않으면 강제로 끝낸다."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def format_stage_counts(summary, missing=0):
    """단계 집계를 한 줄 문구로 만든다."""
    if summary is None:
        return "집계를 읽지 못했습니다"
    counts = (
        f"전체 {summary['total']}개, 성공 {summary['succeeded']}개, "
        f"건너뜀 {summary['skipped']}개, 실패 {summary['failed']}개"
    )
    # 샤드 실행에서는 1단계 경고가 로그 파일로만 가므로 여기서 다시 보여 준다.
    short = summary.get("short_transcripts")
    if short:
        counts = f"{counts}, 뒷부분 누락 의심 {short}개"
    if missing:
        return f"{counts} (샤드 {missing}개의 집계를 읽지 못해 실제보다 적음)"
    return counts


def merge_stage_code(worst, code):
    """단계 종료 코드를 지금까지의 최악값과 합친다.

    신호로 죽은 자식(OOM 킬러 등)은 음수라 max()로 가장 좋은 코드처럼 보이므로,
    예상 밖의 코드는 2로 올려 잡는다.
    """
    if code not in (0, 1, 2):
        return 2
    return max(worst, code)


def describe_exit(code):
    if code == 0:
        return "전부 성공"
    if code == 1:
        return "일부 파일 실패"
    if code == 2:
        return "폴더 또는 모델 경로 문제"
    return f"알 수 없는 오류(코드 {code})"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="STT → 화자분리 → 익명화 독립 단계 실행 파이프라인",
        epilog="-- 뒤에 지정한 인자는 1단계(stt_server.py)에 직접 전달됩니다.",
    )
    parser.add_argument(
        "--wav-folder",
        default=WAV_FOLDER,
        help="1단계 입력 WAV 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--text-folder",
        default=TEXT_FOLDER,
        help="1단계 출력 = 3단계 입력 TXT 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--timestamps-folder",
        default=STT_TIMESTAMPS_FOLDER,
        help="1단계 단어·구간 타임스탬프 JSON 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--diarization-folder",
        default=DIARIZATION_FOLDER,
        help="2단계 화자분리 전사 TXT 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--stt-cache-folder",
        default=STT_CACHE_FOLDER,
        help="1단계 재개 상태 JSON 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--result-folder",
        default=RESULT_FOLDER,
        help="3단계 익명화 결과 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--info-folder",
        default=INFO_FOLDER,
        help="3단계 익명화 기록(JSON) 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--diarization-result-folder",
        default=DIARIZATION_RESULT_FOLDER,
        help="4단계 화자분리 전사 익명화 결과 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--registry-file",
        help="3·4단계 사전 치환 목록 JSON 경로 (기본값: anonymization_registry.json)",
    )
    parser.add_argument(
        "--diarization-model",
        help=(
            "2단계 pyannote 파이프라인 폴더/config.yaml 경로 또는 허브 ID "
            "(생략하면 diarization_server.py의 기본값과 "
            "DIARIZATION_MODEL_PATH 환경변수를 따릅니다)"
        ),
    )
    parser.add_argument("--num-speakers", type=int, help="화자 수 고정 (기본값: 2)")
    parser.add_argument("--min-speakers", type=int, help="화자 수 하한")
    parser.add_argument("--max-speakers", type=int, help="화자 수 상한")
    parser.add_argument(
        "--sample",
        type=int,
        help="샘플 실행: 각 단계에서 경로순 상위 N개만 처리",
    )
    parser.add_argument(
        "--gpus",
        help=(
            "여러 GPU에 파일을 나눠 동시 처리 "
            "(예: --gpus 0,1,2,3 또는 --gpus auto). "
            "지정하면 각 단계가 카드마다 별도 프로세스로 돌고, "
            "출력은 [GPU 2] 같은 이름표를 붙여 터미널에 그대로 나오면서 "
            "--log-folder에 샤드별 로그로도 남습니다"
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "카드 한 장에 띄울 STT 프로세스 수 (기본값: %(default)s). "
            "화자분리와 익명화는 항상 카드당 1개입니다"
        ),
    )
    parser.add_argument(
        "--log-folder",
        default=LOG_FOLDER,
        help="분산 실행 시 샤드별 로그 폴더 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--quiet-shards",
        dest="stream_shards",
        action="store_false",
        help=(
            "샤드 출력을 터미널에 흘리지 않고 --log-folder의 로그 파일에만 "
            "씁니다. 여러 카드의 줄이 섞이는 것이 거슬릴 때 씁니다"
        ),
    )
    parser.add_argument(
        "--skip-stt",
        action="store_true",
        help="Whisper STT를 건너뛰고 기존 타임스탬프로 화자분리부터 실행",
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="화자분리 단계를 건너뜀",
    )
    parser.add_argument(
        "--skip-anonymize",
        action="store_true",
        help="익명화 단계를 건너뜀 (3단계와 4단계 모두)",
    )
    parser.add_argument(
        "--skip-diarization-anonymize",
        action="store_true",
        help=(
            "4단계만 건너뜀. Diarization/에 익명화하지 않은 전사가 그대로 "
            "남습니다"
        ),
    )
    parser.add_argument(
        "--stop-on-partial",
        action="store_true",
        help="STT에서 일부 파일 실패 시 후속 단계를 중단",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "같은 상대 경로에 기존 결과 TXT가 있으면 해당 파일의 추론을 건너뜀 "
            "(STT는 Text/와 STT-Timestamps/, 화자분리는 Diarization/, "
            "익명화는 Anonymization/와 Anonymization_diar/ 기준)"
        ),
    )
    parser.add_argument(
        "stt_args",
        nargs="*",
        help="stt_server.py로 넘길 추가 인자 (-- 뒤에 지정)",
    )

    args = parser.parse_args(argv)

    if args.skip_stt and args.skip_diarization and args.skip_anonymize:
        parser.error("세 단계를 모두 건너뛰면 할 일이 없습니다.")
    if args.skip_stt and args.stt_args:
        parser.error("--skip-stt와 stt_server.py 전달 인자를 함께 쓸 수 없습니다.")
    if args.sample is not None and args.sample < 1:
        parser.error(f"--sample은 1 이상이어야 합니다: {args.sample}")
    if args.workers_per_gpu < 1:
        parser.error(
            f"--workers-per-gpu는 1 이상이어야 합니다: {args.workers_per_gpu}"
        )
    if "--shard" in args.stt_args:
        # 부모가 뒤에 붙이는 --shard가 이겨 조용히 무시되므로 미리 막는다.
        parser.error(
            "--shard는 run_pipeline.py가 직접 붙입니다. "
            "나눠 돌리려면 --gpus 또는 --workers-per-gpu를 쓰세요."
        )

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error("--num-speakers는 --min-speakers/--max-speakers와 함께 쓸 수 없습니다.")
    for name, value in (
        ("--num-speakers", args.num_speakers),
        ("--min-speakers", args.min_speakers),
        ("--max-speakers", args.max_speakers),
    ):
        if value is not None and value < 1:
            parser.error(f"{name}는 1 이상이어야 합니다: {value}")

    try:
        args.gpus = resolve_gpus(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def has_txt_files(folder):
    """폴더 아래에 익명화할 TXT가 하나라도 있는지 본다.

    비어 있을 때 4단계를 오류(코드 2)로 끝내면 실패한 실행처럼 보이므로 미리 본다.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() == ".txt"
        for path in root.rglob("*")
    )


def sample_arguments(sample):
    return [] if sample is None else ["--sample", str(sample)]


def resume_arguments(resume):
    return ["--resume"] if resume else []


def registry_arguments(path):
    return [] if path is None else ["--registry-file", path]


def diarization_model_arguments(path):
    return [] if path is None else ["--diarization-model", path]


def speaker_arguments(num_speakers, min_speakers, max_speakers):
    arguments = []
    if num_speakers is not None:
        arguments += ["--num-speakers", str(num_speakers)]
    if min_speakers is not None:
        arguments += ["--min-speakers", str(min_speakers)]
    if max_speakers is not None:
        arguments += ["--max-speakers", str(max_speakers)]
    return arguments


def main(argv=None):
    args = parse_args(argv)

    for script in (STT_SCRIPT, DIARIZATION_SCRIPT, ANONYMIZER_SCRIPT):
        if not script.exists():
            print(f"오류: 스크립트가 없습니다: {script}")
            return 2

    worst = 0
    stt_summary = None
    diarization_summary = None
    anonymize_summary = None
    diarization_anonymize_summary = None
    stt_missing = 0
    diarization_missing = 0
    anonymize_missing = 0
    diarization_anonymize_missing = 0
    stt_ran = False
    diarization_ran = False
    anonymize_ran = False
    diarization_anonymize_ran = False

    def report():
        """단계별 성공·실패 집계를 마지막에 한 번 출력한다."""
        print("\n전체 결과:")
        if args.sample is not None:
            print(f"  샘플 실행: 단계별 최대 {args.sample}개")
        if stt_ran:
            print(
                "  1단계 음성 인식: "
                f"{format_stage_counts(stt_summary, stt_missing)}"
            )
            print(
                f"    {args.wav_folder} → {args.text_folder}, "
                f"{args.timestamps_folder}"
            )
        if diarization_ran:
            print(
                "  2단계 화자분리:   "
                f"{format_stage_counts(diarization_summary, diarization_missing)}"
            )
            print(
                f"    {args.wav_folder} + {args.timestamps_folder} → "
                f"{args.diarization_folder}"
            )
        if anonymize_ran:
            print(
                "  3단계 익명화:   "
                f"{format_stage_counts(anonymize_summary, anonymize_missing)}"
            )
            print(
                f"    {args.text_folder} → {args.result_folder}, "
                f"{args.info_folder}"
            )
        if diarization_anonymize_ran:
            print(
                "  4단계 화자분리 익명화: "
                f"{format_stage_counts(diarization_anonymize_summary, diarization_anonymize_missing)}"
            )
            print(
                f"    {args.diarization_folder} → "
                f"{args.diarization_result_folder} (기록 JSON 없음)"
            )
        print(f"  최종 상태: {describe_exit(worst)}")

    stt_shards = len(args.gpus or [None]) * args.workers_per_gpu
    diarization_shards = len(args.gpus) if args.gpus else 1
    anonymize_shards = len(args.gpus) if args.gpus else 1
    # Diarization/에 TXT가 있는지는 2단계 뒤에 보므로 여기서는 플래그만 본다.
    diarization_anonymize_planned = not (
        args.skip_anonymize or args.skip_diarization_anonymize
    )
    active_shards = []
    if not args.skip_stt:
        active_shards.append(stt_shards)
    if not args.skip_diarization:
        active_shards.append(diarization_shards)
    if not args.skip_anonymize:
        active_shards.append(anonymize_shards)
    if diarization_anonymize_planned:
        active_shards.append(anonymize_shards)
    if max(active_shards) > 1:
        print("분산 실행:")
        if not args.skip_stt:
            print(f"  1단계 {stt_shards}개 샤드")
        if not args.skip_diarization:
            print(f"  2단계 {diarization_shards}개 샤드")
        if not args.skip_anonymize:
            print(f"  3단계 {anonymize_shards}개 샤드")
        if diarization_anonymize_planned:
            print(f"  4단계 {anonymize_shards}개 샤드")
        print(f"  샤드가 둘 이상인 단계는 {args.log_folder}에 로그를 남깁니다.")
        # 익명화는 잘린 결과를 버릴 때 원문 끝부분을 로그에 남길 수 있다.
        print(
            "  로그에 원문 내용이 남을 수 있습니다. "
            "Text/·Anonymization/와 같은 수준으로 보호하거나 "
            "TRUNCATED_TAIL_CHARS=0으로 실행하세요."
        )
    elif args.gpus:
        print(
            f"GPU {args.gpus[0]}번만 사용합니다. "
            "나누지 않으므로 진행 상황은 터미널에 그대로 나옵니다."
        )

    if args.sample is not None:
        print(f"샘플 실행: 각 단계에서 경로순 앞의 {args.sample}개만 처리합니다.")
        if not args.skip_stt and not args.skip_anonymize:
            print(
                "  3단계는 Text 폴더를 다시 훑어 앞의 "
                f"{args.sample}개를 집으므로, 예전 실행이 남긴 파일이 있으면 "
                "1단계가 방금 만든 것과 다를 수 있습니다."
            )
        if diarization_anonymize_planned:
            print(
                "  4단계도 Diarization 폴더를 다시 훑어 앞의 "
                f"{args.sample}개를 집습니다."
            )

    with tempfile.TemporaryDirectory(prefix="pipeline-summary-") as summary_dir:
        try:
            if args.skip_stt:
                print(
                    "1단계를 건너뜁니다. 기존 Text와 STT-Timestamps를 "
                    "그대로 씁니다."
                )
            else:
                stt_ran = True
                stt_code, stt_summary, stt_missing = run_stage(
                    "1단계 음성 인식 (stt_server.py)",
                    STT_SCRIPT,
                    [
                        "--input-folder",
                        args.wav_folder,
                        "--output-folder",
                        args.text_folder,
                        "--timestamps-folder",
                        args.timestamps_folder,
                        "--stt-cache-folder",
                        args.stt_cache_folder,
                        *diarization_model_arguments(args.diarization_model),
                        *sample_arguments(args.sample),
                        *resume_arguments(args.resume),
                        *args.stt_args,
                    ],
                    summary_dir,
                    gpus=args.gpus,
                    workers_per_gpu=args.workers_per_gpu,
                    log_folder=args.log_folder,
                    stream=args.stream_shards,
                )
                worst = merge_stage_code(worst, stt_code)

                if stt_code == 2:
                    print("\n1단계가 시작조차 못 했으므로 후속 단계를 실행하지 않습니다.")
                    return worst
                if stt_code == 1 and args.stop_on_partial:
                    print("\n1단계에서 실패한 파일이 있어 후속 단계를 실행하지 않습니다.")
                    report()
                    return worst
                if stt_code == 1:
                    print(
                        "\n1단계에서 실패한 파일이 있지만, 성공한 파일로 2단계를 계속합니다."
                        " (멈추려면 --stop-on-partial)"
                    )
                if stt_code not in (0, 1, 2):
                    print(
                        f"\n1단계가 비정상 종료했으므로(코드 {stt_code}) "
                        "후속 단계를 실행하지 않습니다."
                    )
                    report()
                    return worst

            if args.skip_diarization:
                print("\n--skip-diarization: 화자분리를 실행하지 않습니다.")
            else:
                diarization_ran = True
                (
                    diarization_code,
                    diarization_summary,
                    diarization_missing,
                ) = run_stage(
                    "2단계 화자분리 (diarization_server.py)",
                    DIARIZATION_SCRIPT,
                    [
                        "--input-folder",
                        args.wav_folder,
                        "--timestamps-folder",
                        args.timestamps_folder,
                        "--output-folder",
                        args.diarization_folder,
                        *sample_arguments(args.sample),
                        *resume_arguments(args.resume),
                        *diarization_model_arguments(args.diarization_model),
                        *speaker_arguments(
                            args.num_speakers,
                            args.min_speakers,
                            args.max_speakers,
                        ),
                    ],
                    summary_dir,
                    gpus=args.gpus,
                    workers_per_gpu=1,
                    log_folder=args.log_folder,
                    stream=args.stream_shards,
                )
                worst = merge_stage_code(worst, diarization_code)
                if diarization_code:
                    print(
                        "\n화자분리에 실패한 파일이 있지만 Text/는 변경되지 않았고, "
                        "익명화는 평문으로 계속할 수 있습니다."
                    )

            if args.skip_anonymize:
                print("\n--skip-anonymize: 익명화를 실행하지 않고 마칩니다.")
                report()
                return worst

            anonymize_ran = True
            anonymize_code, anonymize_summary, anonymize_missing = run_stage(
                "3단계 평문 익명화 (anonimizer_server.py)",
                ANONYMIZER_SCRIPT,
                [
                    "--input-folder",
                    args.text_folder,
                    "--output-folder",
                    args.result_folder,
                    "--info-folder",
                    args.info_folder,
                    "--input-kind",
                    "plain",
                    *sample_arguments(args.sample),
                    *resume_arguments(args.resume),
                    *registry_arguments(args.registry_file),
                ],
                summary_dir,
                gpus=args.gpus,
                # 익명화 모델 하나가 카드 메모리 대부분을 차지해 카드당 1개만 띄운다.
                workers_per_gpu=1,
                log_folder=args.log_folder,
                stream=args.stream_shards,
                summary_name="anonimizer_server.plain",
            )
            worst = merge_stage_code(worst, anonymize_code)

            if args.skip_diarization_anonymize:
                print(
                    "\n--skip-diarization-anonymize: 4단계를 실행하지 않습니다. "
                    f"{args.diarization_folder}에는 익명화하지 않은 전사가 "
                    "그대로 남습니다."
                )
            elif anonymize_code not in (0, 1):
                # 3단계가 시작조차 못 한 이유는 4단계에서도 그대로 재현된다.
                print(
                    f"\n3단계가 시작하지 못했으므로(코드 {anonymize_code}) "
                    "4단계도 실행하지 않습니다."
                )
            elif not has_txt_files(args.diarization_folder):
                print(
                    f"\n{args.diarization_folder}에 익명화할 TXT가 없어 "
                    "4단계를 건너뜁니다."
                )
            else:
                diarization_anonymize_ran = True
                (
                    diarization_anonymize_code,
                    diarization_anonymize_summary,
                    diarization_anonymize_missing,
                ) = run_stage(
                    "4단계 화자분리 전사 익명화 (anonimizer_server.py)",
                    ANONYMIZER_SCRIPT,
                    [
                        "--input-folder",
                        args.diarization_folder,
                        "--output-folder",
                        args.diarization_result_folder,
                        # --no-info라 --info-folder는 넘기지 않는다.
                        "--input-kind",
                        "diarized",
                        "--no-info",
                        *sample_arguments(args.sample),
                        *resume_arguments(args.resume),
                        *registry_arguments(args.registry_file),
                    ],
                    summary_dir,
                    gpus=args.gpus,
                    workers_per_gpu=1,
                    log_folder=args.log_folder,
                    stream=args.stream_shards,
                    summary_name="anonimizer_server.diarized",
                )
                worst = merge_stage_code(worst, diarization_anonymize_code)
                if diarization_anonymize_code:
                    print(
                        "\n4단계에 실패한 파일이 있습니다. 그 파일은 "
                        f"{args.diarization_result_folder}에 결과가 없으므로, "
                        f"{args.diarization_folder}의 원문을 그대로 다루세요."
                    )
        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.")
            return EXIT_INTERRUPTED

    report()
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
