import argparse
import os
from pathlib import Path

import stt_server as stt


SERVER_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", str(SERVER_DIR / "Original"))
TIMESTAMPS_FOLDER = os.environ.get(
    "STT_TIMESTAMPS_FOLDER", str(SERVER_DIR / "STT-Timestamps")
)
OUTPUT_FOLDER = os.environ.get(
    "DIARIZATION_FOLDER", str(SERVER_DIR / "Diarization")
)


def build_jobs(
    input_folder=INPUT_FOLDER,
    timestamps_folder=TIMESTAMPS_FOLDER,
    output_folder=OUTPUT_FOLDER,
    sample=None,
    shard=None,
):
    """(WAV, Whisper 타임스탬프 JSON, 화자분리 TXT) 작업 목록을 만든다."""
    input_dir = Path(input_folder).expanduser().resolve()
    timestamps_dir = Path(timestamps_folder).expanduser().resolve()
    output_dir = Path(output_folder).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"입력 경로는 폴더여야 합니다: {input_dir}")
    if not timestamps_dir.exists():
        raise FileNotFoundError(f"STT 타임스탬프 폴더가 없습니다: {timestamps_dir}")
    if not timestamps_dir.is_dir():
        raise ValueError(f"STT 타임스탬프 경로는 폴더여야 합니다: {timestamps_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"화자분리 출력 경로는 폴더여야 합니다: {output_dir}")

    wav_files = stt.find_wav_files(input_dir)
    if not wav_files:
        raise ValueError(f"입력 폴더에 WAV 파일이 없습니다: {input_dir}")
    wav_files = stt.limit_sources(wav_files, sample, "WAV 파일")
    wav_files = stt.shard_sources(wav_files, shard, "WAV 파일")

    return [
        (
            source,
            timestamps_dir / source.relative_to(input_dir).with_suffix(".json"),
            output_dir / source.relative_to(input_dir).with_suffix(".txt"),
        )
        for source in wav_files
    ]


def process_folder(
    input_folder=INPUT_FOLDER,
    timestamps_folder=TIMESTAMPS_FOLDER,
    output_folder=OUTPUT_FOLDER,
    num_speakers=stt.DEFAULT_NUM_SPEAKERS,
    min_speakers=stt.DEFAULT_MIN_SPEAKERS,
    max_speakers=None,
    sample=None,
    shard=None,
    resume=False,
    ensure_model_loaded=None,
):
    """저장된 Whisper 타임스탬프와 새 pyannote 결과를 정렬한다."""
    jobs = build_jobs(
        input_folder,
        timestamps_folder,
        output_folder,
        sample,
        shard,
    )
    total = len(jobs)
    succeeded = 0
    skipped = 0
    failures = []

    for index, (source_path, timestamp_path, output_path) in enumerate(jobs, start=1):
        if resume and stt.reusable_text_output(source_path, output_path):
            skipped += 1
            print(f"[{index}/{total}] 기존 화자분리 TXT 건너뜀: {output_path}")
            continue

        print(f"[{index}/{total}] 화자분리 중: {source_path}")
        try:
            document = stt.read_timestamp_document(timestamp_path)
            stt.report_recorded_coverage(document)
            chunks, folded = stt.prepare_transcription_chunks(document["chunks"])
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((source_path, str(exc)))
            print(f"  실패: {exc}")
            continue

        # 모델 로드 실패는 개별 파일 문제가 아니므로 호출자로 전파한다.
        if ensure_model_loaded is not None:
            ensure_model_loaded()

        try:
            turns = stt.run_diarization(
                source_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diarized_text = stt.format_prepared_diarized_text(
                chunks, turns, folded
            )
            stt.atomic_write_text(output_path, diarized_text + "\n")
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((source_path, str(exc)))
            print(f"  실패: {exc}")
            continue
        finally:
            stt.clear_gpu_cache()

        succeeded += 1
        print(f"  화자분리 저장 완료: {output_path}")

    return {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="STT 타임스탬프와 pyannote를 정렬해 화자분리 TXT 생성"
    )
    parser.add_argument(
        "--input-folder",
        default=INPUT_FOLDER,
        help="원본 WAV 폴더 (기본값: 실행 파일 폴더/Original)",
    )
    parser.add_argument(
        "--timestamps-folder",
        default=TIMESTAMPS_FOLDER,
        help="stt_server.py가 만든 타임스탬프 JSON 폴더",
    )
    parser.add_argument(
        "--output-folder",
        default=OUTPUT_FOLDER,
        help="화자분리 TXT 폴더 (기본값: 실행 파일 폴더/Diarization)",
    )
    parser.add_argument(
        "--diarization-model",
        default=stt.DIARIZATION_MODEL_PATH,
        help=(
            "pyannote 파이프라인 폴더(안에 config.yaml)나 config.yaml 경로, "
            "또는 허브 ID (기본값: %(default)s, "
            "환경변수 DIARIZATION_MODEL_PATH로도 지정 가능)"
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help=(
            f"화자 수를 정확히 알 때 지정 (기본값: {stt.DEFAULT_NUM_SPEAKERS}, "
            "--min/--max-speakers와 함께 못 씀)"
        ),
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help=(
            "화자 수 하한. 지정하면 화자 수 고정이 풀립니다 "
            f"(독백이면 1, 수를 모르면 {stt.MIN_SPEAKERS})"
        ),
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="화자 수 상한",
    )
    parser.add_argument("--sample", type=int, help="경로순 앞 N개만 처리")
    parser.add_argument("--shard", help="분산 실행용 I/N")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 화자분리 TXT가 비어 있지 않으면 건너뜀",
    )
    parser.add_argument("--summary-json", help="처리 집계를 저장할 JSON 경로")

    args = parser.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        parser.error(f"--sample은 1 이상이어야 합니다: {args.sample}")
    if args.shard is not None:
        try:
            args.shard = stt.parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error("--num-speakers는 --min-speakers/--max-speakers와 함께 쓸 수 없습니다.")
    if (
        args.num_speakers is None
        and args.min_speakers is None
        and args.max_speakers is None
    ):
        args.num_speakers = stt.DEFAULT_NUM_SPEAKERS
        args.min_speakers = stt.DEFAULT_MIN_SPEAKERS
    elif args.num_speakers is None and args.min_speakers is None:
        args.min_speakers = stt.MIN_SPEAKERS

    for name, value in (
        ("--num-speakers", args.num_speakers),
        ("--min-speakers", args.min_speakers),
        ("--max-speakers", args.max_speakers),
    ):
        if value is not None and value < 1:
            parser.error(f"{name}는 1 이상이어야 합니다: {value}")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error(
            f"--min-speakers({args.min_speakers})가 "
            f"--max-speakers({args.max_speakers})보다 큽니다."
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    model_loaded = False

    def ensure_model_loaded():
        nonlocal model_loaded
        if model_loaded:
            return
        stt.load_diarizer(args.diarization_model)
        model_loaded = True

    try:
        # 모델 로드는 첫 파일까지 미루지만 경로 오타는 여기서 먼저 잡는다.
        print(f"화자 분리 모델: {stt.resolve_diarization_source(args.diarization_model)}")
    except RuntimeError as exc:
        print(f"오류: {exc}")
        return 2

    try:
        summary = process_folder(
            input_folder=args.input_folder,
            timestamps_folder=args.timestamps_folder,
            output_folder=args.output_folder,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            sample=args.sample,
            shard=args.shard,
            resume=args.resume,
            ensure_model_loaded=ensure_model_loaded,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        return 2

    if args.summary_json:
        try:
            stt.write_run_summary(args.summary_json, summary)
        except OSError as exc:
            print(f"경고: 처리 결과 요약을 저장하지 못했습니다: {exc}")

    print("\n처리 결과:")
    print(f"  전체: {summary['total']}개")
    print(f"  성공: {summary['succeeded']}개")
    print(f"  건너뜀: {summary['skipped']}개")
    print(f"  실패: {summary['failed']}개")
    print(f"  화자분리 전사: {args.output_folder}")
    if summary["failures"]:
        print("\n실패한 파일:")
        for source_path, reason in summary["failures"]:
            print(f"  - {source_path}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
