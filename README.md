# FlashLabs Chroma (analysis notes)

이 레포는 FlashLabs Chroma 계열 모델을 Hugging Face `transformers`의 `trust_remote_code=True` 방식으로 로드하여, **오디오 입력 → (텍스트) + 오디오 응답** 생성 추론을 실험하는 워크스페이스입니다.

> 참고: 이 레포 자체는 “학습(파인튜닝) 스크립트”가 아니라, 모델 체크포인트를 로드했을 때 동작하는 **구성/전처리/생성(추론) 로직**에 초점이 있습니다.

## 0. 빠른 시작

### 요구사항

- Python 3.11+
- (권장) CUDA 환경

### 설치

```bash
pip install -r requirements.txt
```

### 최소 추론 예제

```python
from IPython.display import Audio
from chroma import ChromaInference

engine = ChromaInference("FlashLabs/Chroma-4B")

result = engine.generate_audio(
    input_audio="example/make_taco.wav",
    speaker_name="lebron_james",
    return_text=True,
    max_new_text_tokens=128,
    max_new_tokens=1000,
)

print(result.text)
Audio(result.audio, rate=result.sample_rate)
```

노트북 예제는 `test.ipynb` / `example.ipynb`를 참고하세요.

# 1. 개요

## 1.1 이 프로젝트가 “가져다 쓴 것” vs “새로 만든 것”

코드 기준으로 구성요소를 분해하면 다음과 같습니다.

### 재사용(외부 구현/사전학습 계열)

- **Qwen2.5-Omni (thinker)**
  - 구현: `transformers.models.qwen2_5_omni.*`
  - 역할: 대화/오디오 입력을 이해하고, 생성 중간에 텍스트 토큰(추론/응답)을 단계적으로 생성
- **Llama 계열(백본/디코더 내부 트랜스포머)**
  - 구현: `transformers.models.llama.modeling_llama.LlamaModel`
  - 역할: 오디오 코드북 토큰을 생성하기 위한 트랜스포머 블록
- **Mimi (codec_model)**
  - 구현: `transformers.models.mimi.modeling_mimi.MimiModel`
  - 역할: 오디오 codebook 토큰 ↔ 파형(24kHz) 변환(encode/decode)

### Chroma에서 커스텀(조립/헤드/생성 로직)

- **조립 모델**: `ChromaForConditionalGeneration`
  - `thinker(Qwen2.5-Omni) + backbone(Llama) + decoder(Llama) + codec(Mimi)`를 하나의 모델로 결합
- **오디오 코드북용 임베딩/헤드**: `ChromaAudioEmbedding`, `ChromaCodebookHead` 등
- **커스텀 생성 루프**: “backbone으로 codebook0 → decoder로 나머지 codebook 채우기”를 프레임 단위로 반복
- **프로세서 확장**: conversation(thinker 입력)과 prompt(보이스 클로닝용 텍스트/오디오)를 분리하여 입력 배치 구성

## 1.2 어떤 모델을 “구조 유지한 채 파인튜닝”했나?

이 레포만으로 학습 레시피(무엇을 freeze 했는지, 어떤 데이터로 얼마나 학습했는지)는 확정할 수 없습니다.
다만 체크포인트(`FlashLabs/Chroma-4B`)는 보통 아래 형태로 이해하는 것이 자연스럽습니다.

- `Qwen2.5-Omni` 계열(thinker) + `Llama` 계열(backbone/decoder) + `Mimi`(codec)를 **결합한 후**, 음성대화/보이스클로닝 목적에 맞게 가중치를 학습/조정한 체크포인트

즉, “코덱만 새로 만들었다”라기보다, **기존 아키텍처들을 조립하고(커스텀), 그 조합을 목적에 맞게 학습한** 형태로 보는 편이 정확합니다.

## 1.3 생성 파이프라인(오디오 생성 시작 시 내부 흐름)

`ChromaInference.generate_audio()`를 호출하면 내부적으로는 대략 아래 단계를 따릅니다.

### 1) 입력 구성 (Processor)

- 입력은 크게 두 축입니다.
  - 대화(conversation): system 텍스트 + user 오디오 입력
  - 레퍼런스(prompt): 보이스 클로닝/스타일을 위한 `prompt_text` + `prompt_audio`
- `chroma/processing_chroma.py`의 `ChromaProcessor`가 위 입력을 모델 배치로 변환합니다.
  - conversation → thinker(Qwen2.5-Omni)용 입력(`thinker_input_ids`, `thinker_input_features` 등)
  - prompt_text → 텍스트 토크나이즈 결과(`input_ids`, `attention_mask`)
  - prompt_audio → 파형 텐서(`input_values`, `input_values_cutoffs`, 24kHz로 리샘플)

### 2) 모델 구성 (ChromaForConditionalGeneration)

`chroma/modeling_chroma.py`의 `ChromaForConditionalGeneration`은 다음 블록을 조합합니다.

- thinker: Qwen2.5-Omni 계열(대화/오디오 이해 및 텍스트 토큰 생성)
- backbone: Llama 계열(오디오 코드북의 첫 번째 codebook 예측)
- decoder: 나머지 codebook(1..N) 토큰을 채워 프레임 완성
- codec_model: Mimi(코드북 토큰 ↔ 실제 파형 변환)

### 3) 생성 루프 (오디오 프레임을 한 칸씩 생성)

커스텀 `generate()`는 반복적으로 다음을 수행합니다.

1. (필요 시) thinker가 다음 텍스트 토큰을 생성하고, 그 임베딩을 backbone 입력에 주입합니다.
2. backbone이 다음 오디오 프레임의 `codebook0` 토큰을 샘플/그리디로 선택합니다.
3. decoder가 `codebook0`를 조건으로 나머지 codebook들을 생성해 **프레임 전체 코드북 벡터**를 완성합니다.
4. 프레임들을 누적하다가 EOS 조건을 만족하면 종료합니다.

### 4) 디코딩 (코드북 → 파형)

누적된 codebook 프레임들을 `codec_model.decode(...)`로 디코드해 최종 파형(기본 24kHz)을 얻습니다.

## 1.4 텍스트 출력(중간 추론 결과) 관련 메모

- 기본 오디오 생성 경로는 “오디오 codebook 시퀀스” 중심이라 텍스트 문자열을 자동으로 반환하지 않습니다.
- 이 워크스페이스의 `ChromaInference.generate_audio(return_text=True)` 옵션은 내부 thinker를 **별도로 generate**해서 `AudioResult.text`에 텍스트를 채웁니다.
  - 주의: 오디오 생성 루프에서 암묵적으로 주입된 텍스트 토큰과 **완전히 동일함을 보장하진 않습니다**(텍스트 토큰을 루프에서 수집/반환하도록 모델 쪽을 수정하면 더 일관된 추출이 가능).

## 1.5 언어(한국어 등) 관련 메모

- 이 코드 경로에는 “언어를 자동 판별/고정하는 명시적인 단계”가 없습니다.
- 텍스트 응답 언어는 주로 system prompt/대화 텍스트 및 모델의 학습 분포에 의해 결정됩니다.
- 텍스트는 자연스러운데 오디오가 이상한 경우, 원인이 `Qwen2.5-Omni(이해)` 단독이라기보다 **음성 생성(codebook/codec) 분포/프롬프트/샘플링**일 가능성도 큽니다.

# 2. 문제 정의

이 문서에서 다루는 문제는 아래와 같습니다.

## 2.1 관찰된 현상

- **한국어 입력 + 한국어 프롬프트(prompt_text/prompt_audio)**에서:
  - 생성된 오디오에 **긴 공백(무음) 구간이 많이 포함**됨
  - 간헐적으로 들리는 구간은 **외국인이 한국어 발음을 하는 듯한 발음/억양**으로 느껴짐
- **영어 입력 + 영어 프롬프트**에서는:
  - 공백이 상대적으로 적고
  - 영어 발음이 자연스러움

## 2.2 문제의 의미(모듈 관점)

Chroma는 “텍스트 토큰”을 최종 산출물로 직접 내기보다는, **오디오 codebook 토큰 시퀀스**를 생성하고(codec decode로 파형화) 음성을 만듭니다.

따라서 “텍스트는 한국어로 괜찮은데 오디오는 이상하다/비어 있다”는 증상은 보통 아래 중 하나로 해석됩니다.

- (가능성 높음) **오디오 codebook 생성(backbone/decoder)이 한국어 발화 분포에서 불안정**
- (가능성 중간) **프롬프트 오디오/텍스트 품질/정렬 문제**(긴 무음, 읽기 스타일 불일치 등)

> 결론적으로, 한국어에서 “공백이 많음 + 발음이 어색함”은 `thinker`보다 **backbone/decoder(+codec)** 쪽에서 먼저 원인을 찾고 해결하는 것이 일반적으로 효율적입니다.

# 3. 전략

## 3.1 한국어 음성 생성 개선 목적 파인튜닝 타깃 우선순위

텍스트는 한국어로 비교적 정상인데 오디오가 무음/발음 붕괴라면, 보통 아래 순서가 비용 대비 효율이 좋습니다.

1. **decoder만 튜닝(LoRA/QLoRA 우선)**
2. 효과가 부족하면 **backbone + decoder**로 확장
3. 여전히 “신호가 깨짐/금속성/아티팩트”가 심하면 그때 **codec(Mimi)**까지 고려(난이도/리스크 큼)
4. `thinker(Qwen2.5-Omni)`는 텍스트가 이미 한국어로 잘 나오면 우선 **freeze 권장**

### 왜 decoder부터?

- decoder는 “프레임 내 나머지 codebook(1..N)”을 채우며, 결과 파형 품질/발음/무음 패턴에 큰 영향을 줍니다.
- 전체 대비 파라미터가 작아(아래 4장에서 수치 참고) **빠르게 실험**할 수 있습니다.

## 3.2 데이터 전략(한국어 학습을 위한 최소 조건)

“한국어를 전혀 모른다” 가정이면, 단순히 한국어 오디오만 많이 넣는 것보다 다음 조건이 중요합니다.

- (필수) **정확한 한국어 텍스트–오디오 정렬**
- (권장) 무음이 과도하지 않은 클린 음성(또는 VAD/trim 적용)
- (권장) 다양한 화자/말하기 속도/감정/발화 길이
- (보이스 클로닝 목적이면) 프롬프트 오디오/텍스트와 본문 발화의 스타일 차이를 너무 크게 만들지 않기

# 4. 분석

## 4.1 가중치(모듈)별 역할과 크기(파라미터 수)

아래 파라미터 수는 `test.ipynb`에서 로드된 체크포인트를 기준으로 **실측**한 값입니다.

| 구성요소 | 역할(요약) | 파라미터 수 |
|---|---|---:|
| thinker (Qwen2.5-Omni) | 입력 오디오/대화 이해 + 텍스트 토큰 생성(중간 추론) | 4.703B |
| backbone (Llama) | 오디오 프레임의 codebook0 예측 | 1.011B |
| decoder (Llama) | 나머지 codebook(1..N) 생성(프레임 완성) | 161.6M |
| codec (Mimi) | codebook ↔ 파형 변환 | 79.3M |
| 합계 |  | 5.922B |

> 메모: 현재 추론용 로드 상태에서는 모든 파라미터가 기본적으로 `requires_grad=True`로 보일 수 있습니다. 실제 학습에서는 “무엇을 freeze/LoRA 적용할지”에 따라 trainable 파라미터 수가 크게 달라집니다.

## 4.2 GPU 메모리 추산

아래는 “파라미터/옵티마 상태만” 대략치입니다(activation/버퍼/파편화 제외).

- decoder만(161.6M): AdamW 상태까지 포함 시 대략 **~1.8–2.4 GB**
- backbone+decoder(1.173B): AdamW 상태까지 포함 시 대략 **~13–17.5 GB**
- 전체(5.922B): AdamW 상태까지 포함 시 대략 **~66–88 GB**

현실적으로는 activation이 추가되므로,

- **LoRA/QLoRA로 decoder 또는 backbone+decoder만** 학습하는 실험은 1× 80GB(A100/H100)에서 시작하는 것이 가장 안전합니다.
- **전체 모델을 AdamW로 풀튜닝**하는 접근은 보통 멀티 GPU(2–8× 80GB) + ZeRO/offload 같은 분산 세팅이 필요합니다.

## 4.3 “진짜 잘 작동하는 모델”을 만들기 위한 로드맵(리소스 포함)

이 절의 목표는 “한 번 개선되는지 보자”가 아니라, **한국어에서도 무음이 적고 발음/억양이 자연스러운 수준의 안정적인 음성 생성 모델**을 만드는 데 필요한 작업을 현실적으로 정리하는 것입니다.

### 4.3.1 품질 목표를 먼저 수치화

학습은 결국 “뭘 좋아졌다고 판단할지”가 있어야 수렴합니다. 최소한 아래 3가지는 잡는 편이 좋습니다.

- **무음/끊김 지표**: 출력 파형에서 VAD 기반 무음 비율(%) / 평균 무음 구간 길이
- **발음/가독성 지표**: 한국어 ASR로 WER/CER(동일한 평가 문장/도메인으로 고정)
- **화자/스타일 유지(보이스 클로닝)**: speaker embedding cosine(레퍼런스 vs 출력), 또는 AB 선호도

> MOS 같은 주관 평가는 가장 강력하지만 비용이 크니, 초기엔 “무음 비율 + ASR + AB 소수 샘플” 조합으로도 충분히 방향을 잡을 수 있습니다.

### 4.3.2 데이터 규모/구성(한국어에서 성능이 ‘괜찮다’의 최소 조건)

한국어 텍스트가 정상인데 오디오가 무너지면, 대부분은 **오디오 codebook 생성 분포(=backbone/decoder)가 한국어 발화 특성을 충분히 학습하지 못한 상태**입니다. 이 경우 데이터는 “양”보다 **정렬/클린/다양성**이 더 먼저입니다.

- **정렬(필수)**: 문장 단위 텍스트–오디오 타임라인이 크게 틀어지면, 무음/발음 붕괴가 거의 항상 따라옵니다.
- **클린(권장)**: 긴 무음/잡음/음량 편차를 VAD/trim, loudness normalization 등으로 정리
- **다양성(권장)**: 다화자/속도/억양/감정/도메인(뉴스/대화/내레이션) 편중 완화
- **보이스 클로닝 목적**이라면:
  - 화자당 **30–120분** 수준의 클린 레퍼런스를 확보하는 편이 안정적(짧으면 과적합/무음 복제 위험 증가)
  - prompt_text와 실제 발화 스타일(말투/속도/감정)이 너무 다르면 품질이 흔들립니다.

### 4.3.3 토큰 규모 감(오디오)

코덱 설정상 오디오 토큰은 대략 다음 규모로 생각할 수 있습니다.

- frame_rate ≈ 12.5 fps, codebook 8개 → 초당 약 100 codebook 토큰
- 1시간 음성 ≈ 360,000 오디오 토큰

이 값은 “데이터가 커질수록 학습 시간이 늘어나는 정도”를 감 잡는 용도이며, 실제 속도는 구현/배치/길이/체크포인팅/정밀도에 크게 좌우됩니다.

### 4.3.4 학습 단계(권장 순서)

1) **데이터 파이프라인 고정(가장 중요)**
- 텍스트 정규화(숫자/단위/기호), VAD/trim, loudness norm, 클리핑 제거, 샘플레이트 일관화
- train/valid/test를 “화자/문장/도메인” 기준으로 누수 없이 분리

2) **decoder LoRA/QLoRA로 한국어 발화 안정화**
- 목표: 무음 패턴 감소, 발음/리듬 안정화(큰 파손을 먼저 막는 단계)
- 장점: 파라미터가 작아 반복 실험이 빠르고, 망가져도 롤백이 쉽습니다.

3) **backbone + decoder로 확장(대부분 여기서 품질이 ‘한 단계’ 올라감)**
- 목표: codebook0(백본)부터 한국어 발화 분포에 맞춰 “프레임 전체”의 안정성을 끌어올림
- 비용은 늘지만, 한국어에서만 무너지는 문제에는 보통 decoder 단독보다 효과가 큽니다.

4) **codec(Mimi) 튜닝은 최후순위**
- “금속성/아티팩트/대역 손상” 같은 신호 자체의 문제가 뚜렷할 때만 고려
- 난이도와 리스크가 크므로, 먼저 backbone/decoder로 가능한 한 해결하는 편이 안전합니다.

### 4.3.5 GPU/시간 예산

아래는 “그냥 괜찮은 모델”을 목표로 할 때의 **권장 사양**입니다.

- **권장 시작 하드웨어**: 2× 80GB(H100)
  - decoder 또는 backbone+decoder를 LoRA/QLoRA로 돌리며, valid 지표(무음/ASR/AB)로 반복 개선
- **목표를 상용/프로덕션 수준으로 끌어올릴 때**:
  - (데이터가 커지거나, 더 긴 컨텍스트/큰 배치가 필요하면) 4× 80GB 이상이 유리
  - 전체 풀튜닝은 보통 8~16× 80GB + ZeRO/FSDP 같은 분산 세팅이 필요합니다.

> 핵심: “몇 시간만 돌려서 해결”이라기보다, (데이터 품질이 확보된다는 전제하에) **수십~수백 시간 규모의 한국어 음성 + 반복적인 평가/개선 루프**가 필요합니다. 즉, 리소스는 ‘임대 시간’보다 **데이터/평가 파이프라인을 고정하고 지속적으로 개선할 수 있는 운영 가능성**이 성패를 좌우합니다.


