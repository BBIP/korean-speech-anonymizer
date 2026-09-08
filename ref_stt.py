from pathlib import Path
import inspect
import os

# ## 수정 ##
from faster_whisper import WhisperModel, decode_audio
import torch
import whisper_diarization.diarization.msdd.msdd as msdd

from whisper_diarization.helpers import (
    get_realigned_ws_mapping_with_punctuation,
)

# NeMo INFO/WARNING 로그 억제
import logging
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(logging.ERROR)
# ## 수정 ##

import argparse
import traceback
import time


# ## 수정 ##
MODEL_PATH = os.environ.get(
    "WHISPER_MODEL_PATH",
    "/모델주소/whisper-large-v3-ct2",
)
# ## 수정 ##

AUDIO_EXTENSIONS = {
    ".wav",
}

# ## 수정 ##
# Pass 1과 missing-speech retry가 같은 문구 집합을 사용하게 한 곳에서 관리합니다.
# 자주 나오는 문구를 넣으면 그 표현의 인식률이 올라갑니다.
# 아래는 예시이므로 실제 데이터에 맞는 문구로 교체하세요.
HOTWORDS = " ".join(
    (
        "안녕하세요",
        "네 알겠습니다",
    )
)
# ## 수정 ##


# ============================================================
# ## 수정 ##
# MSDD VAD 설정
# 현재까지 테스트해서 확정한 값
# ============================================================

original_create_config = msdd.create_config


def custom_create_config():
    config = original_create_config()

    config.diarizer.vad.parameters.onset = 0.3
    config.diarizer.vad.parameters.offset = 0.4

    config.diarizer.vad.parameters.pad_onset = 0.3
    config.diarizer.vad.parameters.pad_offset = 0.1

    return config


msdd.create_config = custom_create_config
# ## 수정 ##


# ============================================================
# ## 수정 ##
# whisper-diarization의 2화자 고정 API 확인
# ============================================================

def require_two_speaker_msdd_api(diarizer):
    """현재 파이프라인에 필요한 num_speakers 인자 지원 여부를 확인합니다."""
    parameters = inspect.signature(
        diarizer.diarize
    ).parameters.values()

    if not any(
        parameter.name == "num_speakers"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        raise RuntimeError(
            "현재 whisper_diarization MSDDDiarizer가 num_speakers를 "
            "지원하지 않습니다. 함께 제공한 서버 이전 안내서의 "
            "2화자 패치를 적용하세요."
        )


# ## 수정 ##


# ============================================================
# ## 수정 ##
# Missing Speech Recovery 설정
# ============================================================

# MSDD에서는 speech인데 Whisper word가 이 시간 이상 없으면 재전사
MIN_MISSING_SPEECH_SEC = 1.0

# 누락 구간 재전사 시 앞뒤 문맥 포함
RETRY_PADDING_SEC = 0.5

# 하나의 retry chunk 최대 길이
MAX_RETRY_CHUNK_SEC = 20.0

# Whisper word timestamp의 작은 오차 허용
WORD_COVERAGE_PAD_SEC = 0.15

# 기본 retry 반복 횟수
DEFAULT_RETRY_LOOPS = 1

# retry에서 이 확률보다 낮은 단어는 최종 전사에 추가X
MIN_RETRY_WORD_PROBABILITY = 0.35

# ## 수정 ##


def format_srt_time(seconds):
    ms = int(seconds * 1000)

    h = ms // 3_600_000
    ms %= 3_600_000

    m = ms // 60_000
    ms %= 60_000

    s = ms // 1000
    ms %= 1000

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ============================================================
# ## 수정 ##
# Faster-Whisper segment → word list
# ============================================================

def extract_words_from_segments(segments):

    words = []

    for segment in segments:

        if segment.words is None:
            continue

        for word in segment.words:

            if word.start is None or word.end is None:
                continue

            text = word.word.strip()

            if not text:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": getattr(
                        word,
                        "probability",
                        None,
                    ),
                    "source": "pass1",
                }
            )

    return words


# ============================================================
# ## 수정 ##
# 시간 구간 병합
# ============================================================

def merge_intervals(intervals, merge_gap=0.0):

    if not intervals:
        return []

    intervals = sorted(
        intervals,
        key=lambda x: x[0],
    )

    merged = [
        [intervals[0][0], intervals[0][1]]
    ]

    for start, end in intervals[1:]:

        prev_start, prev_end = merged[-1]

        if start <= prev_end + merge_gap:

            merged[-1][1] = max(
                prev_end,
                end,
            )

        else:

            merged.append(
                [start, end]
            )

    return [
        (start, end)
        for start, end in merged
    ]


# ============================================================
# ## 수정 ##
# MSDD Speaker 0/1 구간 → 전체 speech union
# ============================================================

def get_msdd_speech_intervals(
    speaker_segments,
):

    intervals = []

    for start_ms, end_ms, speaker in speaker_segments:

        intervals.append(
            (
                start_ms / 1000.0,
                end_ms / 1000.0,
            )
        )

    return merge_intervals(
        intervals,
        merge_gap=0.10,
    )


# ============================================================
# ## 수정 ##
# Whisper word가 커버하는 시간 구간
# ============================================================

def get_word_coverage_intervals(words):

    intervals = []

    for word in words:

        start = max(
            0.0,
            word["start"]
            - WORD_COVERAGE_PAD_SEC,
        )

        end = (
            word["end"]
            + WORD_COVERAGE_PAD_SEC
        )

        intervals.append(
            (start, end)
        )

    return merge_intervals(
        intervals,
        merge_gap=0.05,
    )


# ============================================================
# ## 수정 ##
# MSDD는 speech인데 Whisper word가 없는 구간 탐색
# ============================================================

def find_missing_speech_intervals(
    speaker_segments,
    words,
):

    if not speaker_segments:
        return []

    speech_intervals = (
        get_msdd_speech_intervals(
            speaker_segments
        )
    )

    word_intervals = (
        get_word_coverage_intervals(
            words
        )
    )

    # Whisper word 자체가 하나도 없는 경우
    if not word_intervals:

        return [
            (start, end)
            for start, end in speech_intervals
            if (
                end - start
                >= MIN_MISSING_SPEECH_SEC
            )
        ]

    missing = []

    word_idx = 0

    for speech_start, speech_end in speech_intervals:

        cursor = speech_start

        while (
            word_idx < len(word_intervals)
            and word_intervals[word_idx][1]
            <= speech_start
        ):
            word_idx += 1

        idx = word_idx

        while (
            idx < len(word_intervals)
            and word_intervals[idx][0]
            < speech_end
        ):

            word_start, word_end = (
                word_intervals[idx]
            )

            # 현재 speech 영역에 실제로 겹치는 word만 사용
            if word_end <= speech_start:
                idx += 1
                continue

            # word가 나오기 전 speech 영역
            if word_start > cursor:

                gap_start = cursor

                gap_end = min(
                    word_start,
                    speech_end,
                )

                if (
                    gap_end - gap_start
                    >= MIN_MISSING_SPEECH_SEC
                ):

                    missing.append(
                        (
                            gap_start,
                            gap_end,
                        )
                    )

            cursor = max(
                cursor,
                word_end,
            )

            if cursor >= speech_end:
                break

            idx += 1

        # 해당 MSDD speech의 끝부분이 전사되지 않은 경우
        if (
            speech_end - cursor
            >= MIN_MISSING_SPEECH_SEC
        ):

            missing.append(
                (
                    cursor,
                    speech_end,
                )
            )

    return merge_intervals(
        missing,
        merge_gap=0.10,
    )


# ============================================================
# ## 수정 ##
# 긴 Missing Speech 영역을 최대 20초로 분할
# ============================================================

def split_missing_intervals(intervals):

    chunks = []

    for start, end in intervals:

        current = start

        while current < end:

            chunk_end = min(
                current
                + MAX_RETRY_CHUNK_SEC,
                end,
            )

            chunks.append(
                (
                    current,
                    chunk_end,
                )
            )

            current = chunk_end

    return chunks


# ============================================================
# ## 수정 ##
# 누락된 speech 구간만 Faster-Whisper 재전사
# ============================================================

def retry_missing_speech(
    model,
    audio,
    missing_intervals,
):

    recovered_words = []

    audio_duration = (
        len(audio) / 16000.0
    )

    retry_chunks = (
        split_missing_intervals(
            missing_intervals
        )
    )

    for (
        missing_start,
        missing_end,
    ) in retry_chunks:

        retry_start = max(
            0.0,
            missing_start
            - RETRY_PADDING_SEC,
        )

        retry_end = min(
            audio_duration,
            missing_end
            + RETRY_PADDING_SEC,
        )

        start_sample = int(
            retry_start * 16000
        )

        end_sample = int(
            retry_end * 16000
        )

        retry_audio = audio[
            start_sample:end_sample
        ]

        if len(retry_audio) == 0:
            continue

        print(
            f"    STT RETRY "
            f"{missing_start:.2f}"
            f"~{missing_end:.2f}s "
            f"(input "
            f"{retry_start:.2f}"
            f"~{retry_end:.2f}s)"
        )

        retry_segments, _ = (
            model.transcribe(
                retry_audio,

                language="ko",
                task="transcribe",

                beam_size=5,
                temperature=0.0,

                condition_on_previous_text=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,

                # 기존 STT 설정 유지
                vad_filter=False,
                no_speech_threshold=0.6,

                word_timestamps=True,
                # ## 수정 ##
                hotwords=None,
                # ## 수정 ##
            )
        )

        retry_segments = list(
            retry_segments
        )

        for segment in retry_segments:

            if segment.words is None:
                continue

            for word in segment.words:

                if (
                    word.start is None
                    or word.end is None
                ):
                    continue

                # ## 수정 ##
                probability = getattr(
                    word,
                    "probability",
                    None,
                )

                # 확률 정보가 있고 기준보다 낮으면 retry 결과에서 제외
                if (
                    probability is not None
                    and probability
                    < MIN_RETRY_WORD_PROBABILITY
                ):
                    continue
                # ## 수정 ##

                text = word.word.strip()

                if not text:
                    continue

                # retry chunk 내부 timestamp를
                # 원본 오디오 timestamp로 변환
                global_start = (
                    retry_start
                    + float(word.start)
                )

                global_end = (
                    retry_start
                    + float(word.end)
                )

                midpoint = (
                    global_start
                    + global_end
                ) / 2

                # padding 부분에서 기존 문장을 다시 읽은 경우 제외
                if not (
                    missing_start
                    <= midpoint
                    <= missing_end
                ):
                    continue

                recovered_words.append(
                    {
                        "word": text,
                        "start": global_start,
                        "end": global_end,
                        "probability": probability,
                        "source": "retry",
                    }
                )

    return recovered_words


# ============================================================
# ## 수정 ##
# 기존 word + retry word 병합
# ============================================================

def merge_transcription_words(
    original_words,
    recovered_words,
):

    result = [
        word.copy()
        for word in original_words
    ]

    for recovered in recovered_words:

        duplicate = False

        recovered_center = (
            recovered["start"]
            + recovered["end"]
        ) / 2

        for existing in result:

            # 거의 동일한 timestamp에
            # 동일한 단어가 이미 존재하면 duplicate
            if (
                existing["word"]
                != recovered["word"]
            ):
                continue

            existing_center = (
                existing["start"]
                + existing["end"]
            ) / 2

            if (
                abs(
                    existing_center
                    - recovered_center
                )
                <= 0.50
            ):

                duplicate = True
                break

        if not duplicate:

            result.append(
                recovered
            )

    result.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    return result


# ============================================================
# ## 수정 ##
# Whisper word ↔ MSDD speaker overlap 계산
# ============================================================

def get_word_speaker(
    word_start,
    word_end,
    speaker_segments,
):

    word_start_ms = (
        word_start * 1000
    )

    word_end_ms = (
        word_end * 1000
    )

    best_speaker = None
    best_overlap = 0

    for (
        spk_start,
        spk_end,
        speaker,
    ) in speaker_segments:

        overlap = max(
            0,
            min(
                word_end_ms,
                spk_end,
            )
            -
            max(
                word_start_ms,
                spk_start,
            )
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_speaker = speaker

    return (
        best_speaker,
        best_overlap,
    )


# ============================================================
# ## 수정 ##
# overlap이 없는 word는 최종 transcript에 한해
# 가장 가까운 speaker로 보완
# ============================================================

def get_nearest_speaker(
    word_start_ms,
    word_end_ms,
    speaker_segments,
):

    if not speaker_segments:
        return None

    if word_end_ms > word_start_ms:

        anchor = (
            word_start_ms
            + word_end_ms
        ) / 2

    else:

        # Faster-Whisper에서 0-length word가
        # 나오는 경우 처리
        anchor = word_start_ms

    best_speaker = None
    best_distance = float("inf")

    for (
        spk_start,
        spk_end,
        speaker,
    ) in speaker_segments:

        if (
            spk_start
            <= anchor
            <= spk_end
        ):

            return speaker

        distance = min(
            abs(
                anchor
                - spk_start
            ),
            abs(
                anchor
                - spk_end
            ),
        )

        if distance < best_distance:

            best_distance = distance
            best_speaker = speaker

    return best_speaker


# ============================================================
# ## 수정 ##
# 최종 word list → Speaker mapping
# ============================================================

def map_words_to_speakers(
    words,
    speaker_segments,
):

    raw_words = []

    total_words = 0
    unassigned_words = 0

    for word in words:

        text = word["word"]

        if not text:
            continue

        total_words += 1

        speaker, overlap_ms = (
            get_word_speaker(
                word["start"],
                word["end"],
                speaker_segments,
            )
        )

        # RAW 기준 QC
        if speaker is None:
            unassigned_words += 1

        raw_words.append(
            {
                "word": text,

                "start_time": int(
                    round(
                        word["start"]
                        * 1000
                    )
                ),

                "end_time": int(
                    round(
                        word["end"]
                        * 1000
                    )
                ),

                "speaker": speaker,
                "overlap_ms": overlap_ms,

                "source": word.get(
                    "source",
                    "unknown",
                ),
            }
        )

    # --------------------------------------------------------
    # 최종 transcript용 mapping
    # RAW QC 데이터와 별도로 처리
    # --------------------------------------------------------

    final_words = []

    for item in raw_words:

        new_item = item.copy()

        if new_item["speaker"] is None:

            new_item["speaker"] = (
                get_nearest_speaker(
                    new_item[
                        "start_time"
                    ],
                    new_item[
                        "end_time"
                    ],
                    speaker_segments,
                )
            )

        final_words.append(
            new_item
        )

    # --------------------------------------------------------
    # 문장 중간에 한두 단어만 speaker가 튀는 현상 보정
    # --------------------------------------------------------

    if (
        final_words
        and speaker_segments
    ):

        final_words = (
            get_realigned_ws_mapping_with_punctuation(
                final_words
            )
        )

    return (
        raw_words,
        final_words,
        total_words,
        unassigned_words,
    )


# ============================================================
# ## 수정 ##
# 같은 speaker의 연속 word → 하나의 발화 turn
# ============================================================

def build_speaker_turns(
    words,
    max_gap_ms=1000,
):

    turns = []

    for word in words:

        speaker = word["speaker"]
        start = word["start_time"]
        end = word["end_time"]
        text = word["word"]

        if not turns:

            turns.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "words": [text],
                }
            )

            continue

        previous = turns[-1]

        gap = (
            start
            - previous["end"]
        )

        if (
            previous["speaker"]
            == speaker
            and gap <= max_gap_ms
        ):

            previous["end"] = end

            previous["words"].append(
                text
            )

        else:

            turns.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "words": [text],
                }
            )

    return turns


# ============================================================
# ## 수정 ##
# 파일 하나 STT + Diarization + Missing Speech Recovery
# ============================================================

def transcribe_audio(
    model,
    diarizer,
    input_file,
    txt_file,
    srt_file,
    retry_loops,
):

    # --------------------------------------------------------
    # 오디오 한 번만 디코딩
    # Whisper와 MSDD가 같이 사용
    # --------------------------------------------------------

    audio = decode_audio(
        str(input_file)
    )

    # ========================================================
    # Pass 1: 전체 STT
    # ========================================================

    segments, info = model.transcribe(
        audio,

        language="ko",
        task="transcribe",

        beam_size=5,
        temperature=0.0,

        condition_on_previous_text=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,

        vad_filter=False,
        no_speech_threshold=None,

        # ## 수정 ##
        word_timestamps=True,
        hotwords=HOTWORDS,
        # ## 수정 ##
    )

    segments = list(
        segments
    )

    # ========================================================
    # ## 수정 ##
    # Pass 1 word 추출
    # ========================================================

    pass1_words = (
        extract_words_from_segments(
            segments
        )
    )

    # ========================================================
    # ## 수정 ##
    # MSDD
    # ========================================================

    diarization_status = "ok"

    try:

        speaker_segments = (
            diarizer.diarize(
                torch.from_numpy(
                    audio
                ).unsqueeze(0),

                num_speakers=2,
            )
        )

    except ValueError as e:

        message = str(e)

        # MSDD VAD가 전체 silence로 판단한 경우
        if (
            "contains silence"
            in message
            and
            "aborting next steps"
            in message
        ):

            print(
                f"  WARNING: "
                f"MSDD_VAD_SILENCE "
                f"{input_file.name}"
            )

            speaker_segments = []

            diarization_status = (
                "vad_silence"
            )

        else:

            raise

    # ========================================================
    # ## 수정 ##
    # Missing Speech 반복 복구
    # ========================================================

    final_transcription_words = [
        word.copy()
        for word in pass1_words
    ]

    total_retry_regions = 0

    total_unique_recovered_words = 0

    retry_passes_used = 0

    # retry_loops=0이면 실행되지 않음
    for retry_pass in range(
        1,
        retry_loops + 1,
    ):

        if not speaker_segments:
            break

        # 현재까지 복구된 word를 기준으로
        # missing speech 다시 계산
        missing_intervals = (
            find_missing_speech_intervals(
                speaker_segments,
                final_transcription_words,
            )
        )

        if not missing_intervals:

            print(
                f"  Retry pass "
                f"{retry_pass}: "
                f"no missing speech"
            )

            break

        print(
            f"  Retry pass "
            f"{retry_pass}/"
            f"{retry_loops}: "
            f"{len(missing_intervals)} "
            f"missing regions"
        )

        total_retry_regions += len(
            missing_intervals
        )

        retry_passes_used += 1

        recovered_words = (
            retry_missing_speech(
                model,
                audio,
                missing_intervals,
            )
        )

        if not recovered_words:

            print(
                f"  Retry pass "
                f"{retry_pass}: "
                f"recovered 0 words "
                f"-> stop"
            )

            break

        previous_word_count = len(
            final_transcription_words
        )

        final_transcription_words = (
            merge_transcription_words(
                final_transcription_words,
                recovered_words,
            )
        )

        new_word_count = (
            len(
                final_transcription_words
            )
            -
            previous_word_count
        )

        print(
            f"  Retry pass "
            f"{retry_pass}: "
            f"{len(recovered_words)} "
            f"candidate words, "
            f"{new_word_count} "
            f"unique words added"
        )

        # 실제로 새 단어가 추가되지 않았다면
        # 동일 영역을 계속 반복할 필요 없음
        if new_word_count <= 0:

            print(
                f"  Retry pass "
                f"{retry_pass}: "
                f"no unique words "
                f"-> stop"
            )

            break

        total_unique_recovered_words += (
            new_word_count
        )

    # ========================================================
    # ## 수정 ##
    # 모든 retry 종료 후에도 남은 Missing Speech
    # ========================================================

    if speaker_segments:

        remaining_missing_intervals = (
            find_missing_speech_intervals(
                speaker_segments,
                final_transcription_words,
            )
        )

    else:

        remaining_missing_intervals = []

    # ========================================================
    # ## 수정 ##
    # 최종 word → Speaker
    # ========================================================

    (
        raw_words,
        final_words,
        total_words,
        unassigned_words,
    ) = map_words_to_speakers(
        final_transcription_words,
        speaker_segments,
    )

    # ========================================================
    # ## 수정 ##
    # Speaker turn
    # ========================================================

    turns = build_speaker_turns(
        final_words
    )

    txt_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # ## 수정 ##
    # TXT 저장
    # ========================================================

    with open(
        txt_file,
        "w",
        encoding="utf-8",
    ) as f:

        for turn in turns:

            if turn["speaker"] is None:

                speaker_text = (
                    "UNASSIGNED"
                )

            else:

                speaker_text = (
                    f"Speaker "
                    f"{turn['speaker']}"
                )

            text = " ".join(
                turn["words"]
            ).strip()

            if text:

                f.write(
                    f"{speaker_text}: "
                    f"{text}\n"
                )

    # ========================================================
    # ## 수정 ##
    # SRT 저장
    # ========================================================

    with open(
        srt_file,
        "w",
        encoding="utf-8",
    ) as f:

        for idx, turn in enumerate(
            turns,
            start=1,
        ):

            start = format_srt_time(
                turn["start"] / 1000
            )

            end = format_srt_time(
                turn["end"] / 1000
            )

            if turn["speaker"] is None:

                speaker_text = (
                    "UNASSIGNED"
                )

            else:

                speaker_text = (
                    f"Speaker "
                    f"{turn['speaker']}"
                )

            text = " ".join(
                turn["words"]
            ).strip()

            f.write(
                f"{idx}\n"
            )

            f.write(
                f"{start} --> {end}\n"
            )

            f.write(
                f"[{speaker_text}] "
                f"{text}\n\n"
            )

    # ========================================================
    # ## 수정 ##
    # QC 정보
    # ========================================================

    return {
        "segments": len(segments),

        "turns": len(turns),

        "pass1_words": len(
            pass1_words
        ),

        "total_words": (
            total_words
        ),

        "unassigned_words": (
            unassigned_words
        ),

        "diarization_status": (
            diarization_status
        ),

        "retry_passes_used": (
            retry_passes_used
        ),

        "retry_regions": (
            total_retry_regions
        ),

        "recovered_words": (
            total_unique_recovered_words
        ),

        "remaining_missing_regions": (
            len(
                remaining_missing_intervals
            )
        ),
    }


def main(argv=None):

    parser = argparse.ArgumentParser()

    # ## 수정 ##
    parser.add_argument(
        "--input-folder",
        dest="input_dir",
        required=True,
        help="원본 음성 root directory"
    )

    parser.add_argument(
        "--output-folder",
        dest="output_dir",
        required=True,
        help="전사 결과 root directory"
    )

    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="로컬 Faster-Whisper CTranslate2 모델 폴더",
    )

    parser.add_argument(
        "--sample",
        type=int,
        help="경로순 앞 N개 WAV만 처리",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 TXT와 SRT가 모두 있으면 건너뛰고 이어서 처리",
    )
    # ## 수정 ##

    # ========================================================
    # ## 수정 ##
    # 재전사 반복 횟수
    # ========================================================

    parser.add_argument(
        "--retry-loops",
        type=int,
        default=DEFAULT_RETRY_LOOPS,
        help=(
            "MSDD speech 중 STT 누락 구간을 "
            "재전사하는 최대 반복 횟수 "
            "(0=비활성화, 기본값=1)"
        ),
    )

    # ## 수정 ##

    args = parser.parse_args(argv)

    # ## 수정 ##
    if args.retry_loops < 0:

        parser.error(
            "--retry-loops는 "
            "0 이상의 정수여야 합니다."
        )
    if args.sample is not None and args.sample < 1:

        parser.error(
            "--sample은 "
            "1 이상의 정수여야 합니다."
        )
    # ## 수정 ##

    input_root = Path(
        args.input_dir
    ).resolve()

    output_root = Path(
        args.output_dir
    ).resolve()

    # ========================================================
    # ## 수정 ##
    # 입력/출력/모델 경로와 대상 파일을 모델 로드 전에 검증
    # ========================================================

    model_root = Path(
        args.model_path
    ).expanduser().resolve()

    if not input_root.is_dir():

        print(
            f"ERROR: 입력 폴더가 없습니다: "
            f"{input_root}"
        )

        return 2

    if not model_root.is_dir():

        print(
            f"ERROR: 모델 폴더가 없습니다: "
            f"{model_root}"
        )

        return 2

    required_model_files = (
        model_root / "config.json",
        model_root / "model.bin",
    )

    missing_model_files = [
        path.name
        for path in required_model_files
        if not path.is_file()
    ]

    if missing_model_files:

        print(
            "ERROR: Faster-Whisper CTranslate2 모델 파일이 없습니다: "
            + ", ".join(missing_model_files)
            + f" ({model_root})"
        )

        return 2

    if output_root.exists() and not output_root.is_dir():

        print(
            f"ERROR: 출력 경로가 폴더가 아닙니다: "
            f"{output_root}"
        )

        return 2

    try:

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as e:

        print(
            f"ERROR: 출력 폴더를 만들 수 없습니다: "
            f"{output_root}: {e}"
        )

        return 2

    audio_files = sorted(
        (
            path
            for path in input_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in AUDIO_EXTENSIONS
            )
        ),
        key=lambda path: (
            path.relative_to(input_root).as_posix()
        ),
    )

    if not audio_files:

        print(
            f"ERROR: 입력 폴더에 WAV 파일이 없습니다: "
            f"{input_root}"
        )

        return 2

    if args.sample is not None:

        audio_files = audio_files[
            :args.sample
        ]

    # ## 수정 ##

    print("=" * 70)
    print("Model       :", model_root)
    print("Input       :", input_root)
    print("Output      :", output_root)

    # ## 수정 ##
    print(
        "Retry loops :",
        args.retry_loops
    )
    print("Sample      :", args.sample)
    print("Resume      :", args.resume)
    # ## 수정 ##

    print("=" * 70)

    # ========================================================
    # Faster-Whisper
    # ========================================================

    # ## 수정 ##
    try:

        model = WhisperModel(
            str(model_root),
            device="cuda",
            compute_type="float16",
            local_files_only=True,
        )

    except Exception as e:

        print(
            f"ERROR: Faster-Whisper 모델을 불러오지 못했습니다: "
            f"{type(e).__name__}: {e}"
        )

        return 2
    # ## 수정 ##

    # ========================================================
    # ## 수정 ##
    # MSDD도 프로그램 시작 시 1회만 로드
    # ========================================================

    print(
        "Loading MSDD diarizer..."
    )

    # ## 수정 ##
    try:

        require_two_speaker_msdd_api(
            msdd.MSDDDiarizer
        )

        diarizer = (
            msdd.MSDDDiarizer(
                "cuda"
            )
        )

    except Exception as e:

        print(
            f"ERROR: MSDD diarizer를 불러오지 못했습니다: "
            f"{type(e).__name__}: {e}"
        )

        return 2
    # ## 수정 ##

    print(
        "MSDD diarizer loaded."
    )

    # ## 수정 ##

    total = len(
        audio_files
    )

    print(
        f"Found {total} audio files."
    )

    success = 0
    failed = 0
    skipped = 0

    # ========================================================
    # ## 수정 ##
    # 전체 QC
    # ========================================================

    total_words_all = 0
    unassigned_words_all = 0

    total_retry_passes_all = 0
    total_retry_regions_all = 0
    total_recovered_words_all = 0

    vad_silence_count = 0

    # ## 수정 ##

    for idx, input_file in enumerate(
        audio_files,
        start=1,
    ):

        relative = (
            input_file.relative_to(
                input_root
            )
        )

        txt_file = (
            output_root
            / relative
        ).with_suffix(".txt")

        srt_file = (
            output_root
            / relative
        ).with_suffix(".srt")

        # ## 수정 ##
        # --resume일 때만 기존 결과가 둘 다 존재하면 skip
        if (
            args.resume
            and txt_file.exists()
            and srt_file.exists()
        ):

            skipped += 1

            print(
                f"[{idx}/{total}] SKIP"
            )

            continue
        # ## 수정 ##

        start_time = time.time()

        try:

            # =================================================
            # ## 수정 ##
            # retry_loops 전달
            # =================================================

            result = (
                transcribe_audio(
                    model,
                    diarizer,
                    input_file,
                    txt_file,
                    srt_file,
                    args.retry_loops,
                )
            )

            # ## 수정 ##

            elapsed = (
                time.time()
                - start_time
            )

            success += 1

            # =================================================
            # ## 수정 ##
            # 전체 QC 누적
            # =================================================

            total_words_all += (
                result[
                    "total_words"
                ]
            )

            unassigned_words_all += (
                result[
                    "unassigned_words"
                ]
            )

            total_retry_passes_all += (
                result[
                    "retry_passes_used"
                ]
            )

            total_retry_regions_all += (
                result[
                    "retry_regions"
                ]
            )

            total_recovered_words_all += (
                result[
                    "recovered_words"
                ]
            )

            if (
                result[
                    "diarization_status"
                ]
                == "vad_silence"
            ):

                vad_silence_count += 1

            if (
                result["total_words"]
                > 0
            ):

                unassigned_ratio = (
                    result[
                        "unassigned_words"
                    ]
                    /
                    result[
                        "total_words"
                    ]
                    * 100
                )

            else:

                unassigned_ratio = 0.0

            # =================================================
            # ## 수정 ##
            # 파일별 QC 출력
            # =================================================

            status = result[
                "diarization_status"
            ]

            if status == "vad_silence":

                print(
                    f"[{idx}/{total}] "
                    f"WARNING "
                    f"MSDD_VAD_SILENCE "
                    f"segments="
                    f"{result['segments']} "
                    f"words="
                    f"{result['total_words']} "
                    f"unassigned="
                    f"{result['unassigned_words']} "
                    f"time={elapsed:.1f}s"
                )

            else:

                print(
                    f"[{idx}/{total}] OK "
                    f"segments="
                    f"{result['segments']} "
                    f"turns="
                    f"{result['turns']} "
                    f"pass1_words="
                    f"{result['pass1_words']} "
                    f"words="
                    f"{result['total_words']} "
                    f"retry_passes="
                    f"{result['retry_passes_used']} "
                    f"retry_regions="
                    f"{result['retry_regions']} "
                    f"recovered="
                    f"{result['recovered_words']} "
                    f"remaining_missing="
                    f"{result['remaining_missing_regions']} "
                    f"unassigned="
                    f"{result['unassigned_words']} "
                    f"({unassigned_ratio:.2f}%) "
                    f"time={elapsed:.1f}s"
                )

            # ## 수정 ##

        except Exception as e:

            failed += 1

            # ## 수정 ##
            # 재처리가 실패한 파일의 예전/부분 결과가 2단계로 넘어가지 않게 합니다.
            # 성공한 뒤에는 두 파일이 모두 완성되므로 이 경로를 타지 않습니다.
            for incomplete_output in (
                txt_file,
                srt_file,
            ):

                try:

                    incomplete_output.unlink(
                        missing_ok=True,
                    )

                except OSError as cleanup_error:

                    print(
                        "  WARNING: 실패 결과를 지우지 못했습니다: "
                        f"{incomplete_output}: {cleanup_error}"
                    )
            # ## 수정 ##

            print(
                f"[{idx}/{total}] ERROR: "
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

    print()
    print("=" * 70)
    print("Finished")
    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")

    # ========================================================
    # ## 수정 ##
    # 전체 QC 출력
    # ========================================================

    if total_words_all > 0:

        total_unassigned_ratio = (
            unassigned_words_all
            / total_words_all
            * 100
        )

    else:

        total_unassigned_ratio = 0.0

    print(
        f"Words              : "
        f"{total_words_all}"
    )

    print(
        f"Unassigned         : "
        f"{unassigned_words_all} "
        f"({total_unassigned_ratio:.2f}%)"
    )

    print(
        f"Retry passes       : "
        f"{total_retry_passes_all}"
    )

    print(
        f"Retry regions      : "
        f"{total_retry_regions_all}"
    )

    print(
        f"Recovered words    : "
        f"{total_recovered_words_all}"
    )

    print(
        f"MSDD VAD silence   : "
        f"{vad_silence_count}"
    )

    # ## 수정 ##

    print("=" * 70)

    # ## 수정 ##
    return 1 if failed else 0
    # ## 수정 ##


if __name__ == "__main__":
    # ## 수정 ##
    raise SystemExit(main())
    # ## 수정 ##
