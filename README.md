# korean-speech-anonymizer

음성 전사·화자분리·로컬 LLM 기반 한국어 대화 익명화 파이프라인

- 치환 대상: `[이름]`, `[생년월일]`, `[전화번호]`
- 미지원 유형: 이메일, 주소 등

## 처리 단계

```
Original/ ---(1) stt_server.py--------> Text/
          |                             STT-Timestamps/
          |
          +---(2) diarization_server.py-> Diarization/
              (Original/ + STT-Timestamps/ 입력)

Text/ --------(3) anonimizer_server.py-> Anonymization/
                  --input-kind plain     Anonymization_info/

Diarization/ -(4) anonimizer_server.py-> Anonymization_diar/
                  --input-kind diarized
                  --no-info
```

| 단계 | 입력 → 출력 | 비고 |
|---|---|---|
| 1 | 음성 → 평문·단어 타임스탬프 | pyannote 음성 구간 중 누락 단어 재전사 |
| 2 | 음성·타임스탬프 → 화자분리 전사 | 화자 라벨 추가, `Text/` 원본 유지 |
| 3 | 평문 → 익명화 본문·치환 기록 | 치환 전 원문 값의 JSON 기록 |
| 4 | 화자분리 전사 → 익명화 본문 | 별도 치환 기록 생략 |

- 단계별 독립 프로세스 실행 및 종료 시 GPU 메모리 반환
- 2단계 실패 시에도 1단계 평문을 이용한 3단계 진행 가능

## 요구 사항

- Python 3.10 이상
- 파이썬 패키지: `torch`, `transformers`, `pyannote.audio`
- 외부 실행 파일: `ffmpeg`, `ffprobe` (구간 재전사·길이 측정용)
- GPU: CUDA 권장, CPU 실행 시 처리 속도 저하

| 모델 | 용도 | 경로 지정 |
|---|---|---|
| Whisper-large-v3 | 음성 인식 | `MODEL_PATH` (`config.json`이 있는 폴더) |
| pyannote 화자분리 | 화자 라벨과 음성 구간 검출 | `DIARIZATION_MODEL_PATH` (`config.yaml`이 있는 폴더 또는 허브 ID) |
| Instruction 튜닝 LLM | 개인정보 치환 | `MODEL_PATH` (익명화 프로세스 기준) |

기본 경로 `/모델주소/...`는 자리표시자로, 실제 모델 경로 지정 필요

STT와 익명화의 `MODEL_PATH` 환경 변수 공유로 인한 경로 충돌 주의. 전체 파이프라인 실행 전 각 스크립트의 모델 기본값 수정 또는 단계별 환경 변수를 지정한 개별 실행 필요

## 빠른 시작

```bash
# 1. 각 스크립트의 모델 기본값 설정 후 공통 MODEL_PATH 해제
unset MODEL_PATH
export DIARIZATION_MODEL_PATH=/실제/모델/경로/pyannote-community-1

# 2. 원본 WAV 배치
mkdir -p Original && cp /path/to/*.wav Original/

# 3. 전체 4단계 실행
python run_pipeline.py
```

샘플 5건 실행 예시

```bash
python run_pipeline.py --sample 5
```

## 폴더 구조

모든 산출물에서 입력 폴더의 하위 구조 유지

| 폴더 | 내용 | 생성 단계 |
|---|---|---|
| `Original/` | 원본 WAV (입력) | |
| `Text/` | 평문 전사 TXT | 1 |
| `STT-Timestamps/` | 단어 타임스탬프 JSON과 커버리지 판정 | 1 |
| `STT-cache/` | `--resume` 재개 상태 JSON | 1 |
| `Diarization/` | `화자1: 발화` 형식 전사 | 2 |
| `Anonymization/` | 익명화된 평문 | 3 |
| `Anonymization_info/` | 치환 기록 JSON | 3 |
| `Anonymization_diar/` | 익명화된 화자분리 전사 | 4 |
| `logs/` | 분산 실행 시 샤드별 로그 | 1-4 |

치환 기록 JSON 예시

```json
{
  "file": "team-a/call-001.txt",
  "info": {
    "name": ["홍길동"],
    "birthdate": ["1990년 1월 2일"],
    "phone": ["010 1234 5678"]
  }
}
```

## 단계별 스크립트

스크립트별 단독 실행 및 `--help`를 통한 전체 옵션 확인 가능

```bash
python stt_server.py         --help   # 1단계 음성 인식
python diarization_server.py --help   # 2단계 화자분리
python anonimizer_server.py  --help   # 3, 4단계 익명화
python run_pipeline.py       --help   # 전체 실행
```

단독 실행 예시

```bash
# 평문 익명화
python anonimizer_server.py --input-kind plain

# 화자분리 전사 익명화 (기록 JSON 없이)
python anonimizer_server.py --input-kind diarized --no-info \
    --input-folder Diarization --output-folder Anonymization_diar
```

## 주요 옵션

`run_pipeline.py` 기준

| 옵션 | 설명 |
|---|---|
| `--sample N` | 각 단계에서 경로순 앞 N개만 처리 |
| `--resume` | 같은 상대 경로에 결과가 이미 있으면 그 파일의 추론을 건너뜀 |
| `--skip-stt` | 1단계를 건너뛰고 기존 타임스탬프로 2단계부터 실행 |
| `--skip-diarization` | 2단계를 건너뜀 |
| `--skip-anonymize` | 3, 4단계를 건너뜀 |
| `--skip-diarization-anonymize` | 4단계만 건너뜀 |
| `--stop-on-partial` | 1단계에서 일부 파일이 실패하면 후속 단계를 중단 |
| `--num-speakers N` | 화자 수 고정 (기본 2명) |
| `--min-speakers`, `--max-speakers` | 화자 수 하한과 상한 |
| `--registry-file PATH` | 사전 치환 목록 JSON 경로 |
| `--diarization-model PATH` | pyannote 파이프라인 경로 또는 허브 ID |
| `--gpus 0,1,2,3`, `--gpus auto` | 여러 GPU에 파일을 나눠 동시 처리 |
| `--workers-per-gpu N` | GPU당 STT 프로세스 수 |
| `--quiet-shards` | 샤드 출력의 로그 파일 전용 기록 |
| `-- <인자>` | `--` 뒤의 인자는 1단계 `stt_server.py`에 그대로 전달 |

`--resume` 사용 시 원문 변경 여부를 비교하지 않고 기존 결과 재사용. 처리 기준 변경 시 `--resume` 생략 또는 해당 결과 파일 삭제 필요

## 사전 치환 목록

`anonymization_registry.json`: 익명화 시작 시 읽는 사전 치환 목록. 아래 값은 형식 설명용 예시

```json
{
  "schema_version": 1,
  "names": ["홍길동", "김영희"],
  "phone_numbers": ["010-1234-5678", "02-000-0000"]
}
```

등록 값은 추론 전 사전 치환, 빈 목록 사용 시 LLM 기반 치환

실제 값은 `.gitignore` 제외 패턴인 `*.private.json` 파일에 별도 보관

```bash
cp anonymization_registry.json anonymization_registry.private.json
# 복사본에 실제 이름·전화번호 입력 후 실행
python run_pipeline.py --registry-file anonymization_registry.private.json
```

- 이름: 문자열 일치 기준
- 전화번호: 하이픈·공백·점·괄호 표기 차이 허용
- 등록 항목: `[이름]`·`[전화번호]` 치환 및 기록 JSON 포함
- 목록 경로: `--registry-file` 또는 `ANONYMIZATION_REGISTRY_FILE` 지정

## 여러 GPU로 분산 실행

```bash
# 카드 4장에 파일을 나눠 동시 처리
python run_pipeline.py --gpus 0,1,2,3

# nvidia-smi로 감지한 전체 GPU 사용
python run_pipeline.py --gpus auto

# GPU당 STT 프로세스 3개 실행 (화자분리·익명화는 GPU당 1개)
python run_pipeline.py --gpus auto --workers-per-gpu 3
```

- 경로순 파일 목록의 순환 배분
- `[GPU 2]` 형식의 출력 라벨 및 `logs/` 내 샤드별 로그 기록
- `--quiet-shards`: 터미널 출력 생략
- `--workers-per-gpu`: 1단계 STT에만 적용

## 뒷부분 누락 경고

- 경고 기준: 오디오 길이 대비 전사 범위 90% 미만 (`TIMELINE_COVERAGE_MIN_RATIO`)
- 판정 저장: `STT-Timestamps/`의 `coverage` 블록 및 실행 요약
- 종료 코드: 변경 없음, 해당 전사의 뒷부분 누락 여부 확인 필요
- 재처리: 해당 산출물 삭제 후 `--resume` 실행

## 환경 변수

폴더 경로 지정 시 CLI 옵션 우선 적용

### 폴더 경로

| 변수 | 기본값 |
|---|---|
| `WAV_FOLDER`, `INPUT_FOLDER` | `Original` |
| `TEXT_FOLDER`, `OUTPUT_FOLDER` | `Text` |
| `STT_TIMESTAMPS_FOLDER` | `STT-Timestamps` |
| `STT_CACHE_FOLDER` | `STT-cache` |
| `DIARIZATION_FOLDER` | `Diarization` |
| `RESULT_FOLDER` | `Anonymization` |
| `INFO_FOLDER` | `Anonymization_info` |
| `DIARIZATION_RESULT_FOLDER` | `Anonymization_diar` |
| `LOG_FOLDER` | `logs` |
| `ANONYMIZATION_REGISTRY_FILE` | `anonymization_registry.json` |

### 음성 인식 (stt_server.py)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `/모델주소/whisper-large-v3` | Whisper 모델 폴더, 실제 경로 지정 필요 |
| `DIARIZATION_MODEL_PATH` | `/모델주소/pyannote-community-1` | pyannote 파이프라인 폴더 또는 허브 ID |
| `LANGUAGE` | `korean` | 음성 언어 |
| `NUM_SPEAKERS` | `2` | 화자 수 고정. 0이면 하한만 적용 |
| `MIN_SPEAKERS` | `2` | 화자 수 하한 |
| `NO_SPEECH_THRESHOLD` | `0.6` | 무음 판정 기준, 상향 시 환각 증가 가능 |
| `LOGPROB_THRESHOLD` | `-1.0` | 저확률 창 폐기 기준 |
| `COMPRESSION_RATIO_THRESHOLD` | `1.35` | 반복 억제 기준 |
| `REPETITION_PENALTY` | `1.1` | 반복 페널티 |
| `NO_REPEAT_NGRAM_SIZE` | `3` | 반복 금지 n-gram 길이 |
| `TIMELINE_COVERAGE_MIN_RATIO` | `0.9` | 뒷부분 누락 경고 기준 |
| `WORD_CHUNK_SECONDS` | `0` | 0이면 native long-form. 외부 청킹 없음 |
| `MIN_MISSING_SPEECH_SECONDS` | `1.0` | 이 길이 이상 단어가 없는 음성 구간을 재전사 |
| `SPEAKER_MERGE_MAX_GAP` | `1.0` | 줄을 나누는 무음 간격(초) |
| `SPEAKER_MERGE_MAX_CHARS` | `300` | 한 줄 최대 글자 수 |
| `SPEAKER_LABEL_FORMAT` | `화자{index}` | 화자 라벨 형식 |
| `UNKNOWN_SPEAKER_LABEL` | `화자미상` | 화자를 정하지 못한 구간의 라벨 |

### 익명화 (anonimizer_server.py)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `/모델주소/gemma-4-31B-it` | Instruction 튜닝 LLM 폴더, 실제 경로 지정 필요 |
| `MAX_INPUT_TOKENS` | `8192` | 입력 토큰 상한. 넘으면 그 파일은 실패로 처리 |
| `OUTPUT_TOKEN_MARGIN` | `256` | 생성 여유분 고정값 |
| `OUTPUT_TOKEN_MARGIN_RATIO` | `0.5` | 생성 여유분 비례분 |
| `MAX_INFERENCE_SECONDS` | `600` | 추론 시간 상한의 하한선 |
| `INFERENCE_TOKENS_PER_SECOND` | `20` | 시간 상한 계산에 쓰는 예상 생성 속도 |
| `REPETITION_PENALTY` | `1.0` | 반복 페널티 |
| `TRUNCATED_TAIL_CHARS` | `300` | 중단된 결과의 끝부분 로그 길이, 0으로 비활성화 |

## 종료 코드

| 코드 | 의미 |
|---|---|
| 0 | 전부 성공 |
| 1 | 일부 파일 실패. 나머지는 정상 처리 |
| 2 | 폴더 또는 모델 경로 오류로 시작 실패 |
| 130 | 사용자 중단 (Ctrl+C) |

중단 시 자식 프로세스 정리 및 GPU 메모리 반환

## 개인정보 취급 주의

- 원문 포함 폴더: `Text/`, `Diarization/`, `STT-Timestamps/`
- 치환 전 개인정보 포함 폴더: `Anonymization_info/`
- 결과 끝부분의 로그 저장 비활성화: `TRUNCATED_TAIL_CHARS=0` 설정
- 완전한 오프라인 실행: `DIARIZATION_MODEL_PATH`에 `config.yaml` 포함 로컬 폴더 지정 필요
- 신규 데이터·산출물 폴더: `.gitignore` 제외 항목 추가 필요

## 참고 구현

`ref_stt.py`: 기본 파이프라인과 독립된 참고 구현

별도 의존성:

- `faster-whisper` (CTranslate2 변환 모델)
- `whisper-diarization` (NeMo MSDD 화자분리)
