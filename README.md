# korean-speech-anonymizer

한국어 대화 익명화 파이프라인.

음성을 전사하고 화자를 분리한 뒤, 로컬 LLM으로 모든 참여자의 개인정보를 익명화한다.
모델을 로컬 파일에서 불러오므로 음성과 전사를 외부로 보내지 않는다.

치환 대상은 `[이름]`, `[생년월일]`, `[전화번호]`이다. 이메일과 주소 등 다른 개인정보
유형은 현재 지원 범위에 포함되지 않는다.


## 처리 단계

```
Original/ ---(1) stt_server.py--------> Text/
          |                             STT-Timestamps/
          |
          +---(2) diarization_server.py-> Diarization/
              (Original/ + STT-Timestamps/를 함께 읽는다)

Text/ --------(3) anonimizer_server.py-> Anonymization/
                  --input-kind plain     Anonymization_info/

Diarization/ -(4) anonimizer_server.py-> Anonymization_diar/
                  --input-kind diarized
                  --no-info
```

| 단계 | 입력에서 출력으로 | 비고 |
|---|---|---|
| 1 | 음성에서 평문과 단어 타임스탬프로 | pyannote로 검출한 음성 중 단어가 빠진 구간만 다시 전사해 누락을 메운다 |
| 2 | 음성과 타임스탬프에서 화자분리 전사로 | 화자 라벨만 덧붙이며 `Text/`는 건드리지 않는다 |
| 3 | 평문에서 익명화 본문과 치환 기록으로 | 치환된 원문 값이 기록 JSON에 남는다 |
| 4 | 화자분리 전사에서 익명화 본문으로 | 치환 값은 3단계 기록에 이미 있으므로 기록을 남기지 않는다 |

각 단계는 독립 프로세스로 실행된다. GPU 메모리를 단계마다 반납하므로 Whisper와
pyannote와 LLM을 한 프로세스에 함께 올릴 때 생기는 메모리 경쟁이 없다.

2단계가 실패해도 1단계의 평문은 그대로 남아 3단계를 계속 진행할 수 있다.


## 요구 사항

- Python 3.10 이상
- 파이썬 패키지: `torch`, `transformers`, `pyannote.audio`
- 외부 실행 파일: `ffmpeg`, `ffprobe` (구간 재전사와 길이 측정에 쓴다)
- GPU: CUDA 권장. 없으면 CPU로 동작하지만 실용적이지 않다

준비해야 할 로컬 모델은 세 가지다.

| 모델 | 용도 | 경로 지정 |
|---|---|---|
| Whisper-large-v3 | 음성 인식 | `MODEL_PATH` (`config.json`이 있는 폴더) |
| pyannote 화자분리 | 화자 라벨과 음성 구간 검출 | `DIARIZATION_MODEL_PATH` (`config.yaml`이 있는 폴더 또는 허브 ID) |
| Instruction 튜닝 LLM | 개인정보 치환 | `MODEL_PATH` (익명화 프로세스 기준) |

스크립트에 적힌 기본 경로 `/모델주소/...`는 자리표시자다. 환경 변수나 CLI 옵션으로
실제 경로를 지정해야 하며, 지정하지 않으면 모델을 찾지 못해 종료 코드 2로 끝난다.

STT와 익명화가 같은 `MODEL_PATH` 이름을 쓴다. `run_pipeline.py`는 폴더 관련 환경
변수만 걸러내고 나머지는 자식 프로세스에 그대로 물려주므로, `MODEL_PATH`를 셸에 걸어
두면 두 단계가 같은 값을 보게 된다. 두 모델을 함께 지정하려면 각 스크립트를 따로
실행하거나 스크립트의 기본값을 수정한다.


## 빠른 시작

```bash
# 1. 모델 경로를 지정한다
export MODEL_PATH=/실제/모델/경로/whisper-large-v3
export DIARIZATION_MODEL_PATH=/실제/모델/경로/pyannote-community-1

# 2. 원본 WAV를 Original/에 넣는다 (하위 폴더 구조는 그대로 유지된다)
mkdir -p Original && cp /path/to/*.wav Original/

# 3. 전체 4단계 실행
python run_pipeline.py
```

결과는 `Anonymization/`에 익명화된 평문, `Anonymization_info/`에 치환 기록,
`Anonymization_diar/`에 화자 표시가 남아 있는 익명화 본문으로 나온다.

먼저 몇 건만 확인하려면 `--sample`을 쓴다.

```bash
python run_pipeline.py --sample 5
```


## 폴더 구조

입력 폴더의 하위 구조는 모든 산출물에서 그대로 유지된다.

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

치환 기록 JSON은 다음 형태다.

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

각 스크립트는 단독으로도 실행할 수 있다. 전체 옵션은 `--help`에서 확인한다.

```bash
python stt_server.py         --help   # 1단계 음성 인식
python diarization_server.py --help   # 2단계 화자분리
python anonimizer_server.py  --help   # 3, 4단계 익명화
python run_pipeline.py       --help   # 전체 실행
```

단독 실행 예시다.

```bash
# 평문 익명화
python anonimizer_server.py --input-kind plain

# 화자분리 전사 익명화 (기록 JSON 없이)
python anonimizer_server.py --input-kind diarized --no-info \
    --input-folder Diarization --output-folder Anonymization_diar
```


## 주요 옵션

`run_pipeline.py` 기준이다.

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
| `--workers-per-gpu N` | 카드 한 장에 띄울 STT 프로세스 수 |
| `--quiet-shards` | 샤드 출력을 터미널로 흘리지 않고 로그 파일에만 기록 |
| `-- <인자>` | `--` 뒤의 인자는 1단계 `stt_server.py`에 그대로 전달 |

`--resume`은 결과 파일이 비어 있지 않은지만 확인하며 원문과 내용을 비교하지 않는다.
기준을 바꿔 다시 처리하려면 `--resume`을 생략하거나 해당 결과 파일을 지운다.


## 사전 치환 목록

익명화 단계는 시작할 때 사전 치환 목록 파일을 반드시 읽는다. 저장소의
`anonymization_registry.json`은 형식을 보여 주는 예시이며 값은 실제 데이터가 아니다.

```json
{
  "schema_version": 1,
  "names": ["홍길동", "김영희"],
  "phone_numbers": ["010-1234-5678", "02-000-0000"]
}
```

목록이 비어 있어도 LLM이 본문에서 개인정보를 찾아 치환한다. 이 목록은 놓치면 안 되는
값을 추론 전에 확정적으로 치환하기 위한 장치다.

실제 값은 예시 파일에 직접 쓰지 않는다. 별도 파일로 복사해서 쓰면 실수로 커밋할 일이
없다. `*.private.json`은 `.gitignore`가 막아 둔다.

```bash
cp anonymization_registry.json anonymization_registry.private.json
# 복사한 파일에 실제 이름과 전화번호를 채운 뒤
python run_pipeline.py --registry-file anonymization_registry.private.json
```

- 이름은 문자열 일치로 찾는다.
- 전화번호는 하이픈, 공백, 점, 괄호 표기 차이를 허용한다. `010-1234-5678`과
  `010 1234 5678`을 같은 번호로 본다.
- 등록 항목도 일반 항목과 똑같이 `[이름]`과 `[전화번호]`로 치환하고 기록 JSON에 남긴다.
- 다른 목록을 쓰려면 `--registry-file`이나 환경 변수 `ANONYMIZATION_REGISTRY_FILE`을
  지정한다.


## 여러 GPU로 분산 실행

```bash
# 카드 4장에 파일을 나눠 동시 처리
python run_pipeline.py --gpus 0,1,2,3

# nvidia-smi가 보는 카드 전부 사용
python run_pipeline.py --gpus auto

# STT만 카드당 3개씩 겹쳐 올림 (화자분리와 익명화는 항상 카드당 1개)
python run_pipeline.py --gpus auto --workers-per-gpu 3
```

파일은 연속 블록이 아니라 한 칸씩 건너뛰며 나눈다. 경로순 목록에는 길이가 비슷한
녹음이 몰려 있어 블록으로 자르면 한 샤드에만 긴 파일이 쏠린다.

자식 출력에는 `[GPU 2]` 같은 이름표가 붙어 터미널에 그대로 흐르고, 동시에 `logs/`에
샤드별 로그로도 남는다. `--quiet-shards`를 주면 로그 파일에만 기록한다.

익명화 모델 하나가 카드 메모리 대부분을 차지하므로 `--workers-per-gpu`는 1단계에만
적용된다.


## 뒷부분 누락 경고

전사 범위가 오디오 길이의 90퍼센트(`TIMELINE_COVERAGE_MIN_RATIO`)에 못 미치면 경고를
남기고, 판정 결과를 `STT-Timestamps/`의 `coverage` 블록과 실행 요약에 함께 기록한다.
실패가 아니므로 종료 코드는 바뀌지 않지만 해당 TXT는 뒷부분이 비어 있을 수 있다.

문제가 확인된 파일은 산출물을 지우고 `--resume`으로 다시 돌리면 그 파일만 재처리된다.

```bash
python run_pipeline.py --resume
```


## 환경 변수

폴더 경로는 대부분 CLI 옵션으로도 지정할 수 있고, CLI 옵션이 환경 변수보다 우선한다.

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
| `MODEL_PATH` | `/모델주소/whisper-large-v3` | Whisper 모델 폴더. 자리표시자이므로 반드시 지정한다 |
| `DIARIZATION_MODEL_PATH` | `/모델주소/pyannote-community-1` | pyannote 파이프라인 폴더 또는 허브 ID |
| `LANGUAGE` | `korean` | 음성 언어 |
| `NUM_SPEAKERS` | `2` | 화자 수 고정. 0이면 하한만 적용 |
| `MIN_SPEAKERS` | `2` | 화자 수 하한 |
| `NO_SPEECH_THRESHOLD` | `0.6` | 올리면 무음 창을 건너뛰지 않아 환각이 늘어난다 |
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
| `MODEL_PATH` | `/모델주소/gemma-4-31B-it` | Instruction 튜닝 LLM 폴더. 자리표시자이므로 반드시 지정한다 |
| `MAX_INPUT_TOKENS` | `8192` | 입력 토큰 상한. 넘으면 그 파일은 실패로 처리 |
| `OUTPUT_TOKEN_MARGIN` | `256` | 생성 여유분 고정값 |
| `OUTPUT_TOKEN_MARGIN_RATIO` | `0.5` | 생성 여유분 비례분 |
| `MAX_INFERENCE_SECONDS` | `600` | 추론 시간 상한의 하한선 |
| `INFERENCE_TOKENS_PER_SECOND` | `20` | 시간 상한 계산에 쓰는 예상 생성 속도 |
| `REPETITION_PENALTY` | `1.0` | 반복 페널티 |
| `TRUNCATED_TAIL_CHARS` | `300` | 끊긴 결과의 꼬리를 로그에 남길 길이. 0으로 끌 수 있다 |


## 종료 코드

| 코드 | 의미 |
|---|---|
| 0 | 전부 성공 |
| 1 | 일부 파일 실패. 나머지는 정상 처리 |
| 2 | 폴더 또는 모델 경로 문제로 시작하지 못함 |
| 130 | 사용자 중단 (Ctrl+C) |

중단 시 살아 있는 자식 프로세스를 정리한다. 그대로 남기면 GPU 메모리를 붙든 고아가
되어 다음 실행이 out of memory로 실패한다.


## 개인정보 취급 주의

- `Text/`와 `Diarization/`, `STT-Timestamps/`에는 익명화되지 않은 원문이 들어 있다.
  `Anonymization/`만 공유하고 나머지는 같은 수준으로 보호해야 한다.
- 치환 기록 JSON(`Anonymization_info/`)에는 치환 전 원문 값이 담긴다.
- 익명화가 상한에 걸려 끊기면 버려지는 결과의 끝부분을 로그에 남긴다. 분산 실행에서는
  이 내용이 `logs/`의 파일로 그대로 들어가므로, 로그를 남길 때는
  `TRUNCATED_TAIL_CHARS=0`으로 실행한다.
- Whisper와 익명화 LLM은 `local_files_only`로 불러오므로 네트워크를 타지 않는다. 단
  pyannote는 허브 ID를 받을 수 있어, 로컬 폴더 경로가 아니면 모델을 내려받는 과정에서
  네트워크에 접속한다. 완전한 오프라인 실행이 필요하면 `DIARIZATION_MODEL_PATH`에
  `config.yaml`이 있는 로컬 폴더를 지정한다.
- 데이터 폴더와 사전 치환 목록은 `.gitignore`로 제외되어 있다. 새 산출물 폴더를
  추가할 때는 커밋 전에 `.gitignore`에도 함께 넣는다.


## 참고 구현

`ref_stt.py`는 같은 문제를 다른 스택으로 풀어 본 참고용 구현이다. 파이프라인과
연결되어 있지 않고 아래 별도 의존성이 필요하므로 위 4단계 실행에는 쓰이지 않는다.

- `faster-whisper` (CTranslate2 변환 모델)
- `whisper-diarization` (NeMo MSDD 화자분리)
