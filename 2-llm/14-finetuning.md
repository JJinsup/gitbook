---
description: >-
  Unsloth 라이브러리를 활용하여 초경량 LLM인 Qwen 0.6B 모델을 효율적으로 파인튜닝(Fine-tuning)하는 전체 과정을
  다룹니다.
---

# 📚 \[14] 로컬 LLM 파인튜닝&#x20;

{% embed url="https://cloud.google.com/use-cases/fine-tuning-ai-models?hl=ko" %}

### 0. 파인 튜닝(Fine-Tuning)이란?

방대한 데이터로 사전 학습(Pre-training)된 LLM을 **특정 작업이나 도메인에 맞는 데이터셋으로 추가 학습**시키는 과정(전이 학습)입니다.

쉽게 말해, 이미 언어에 대한 일반적인 지식을 갖춘 '똑똑한 신입사원'(사전 학습 모델)에게 "우리 회사의 전문 용어"나 "특정 업무 매뉴얼"을 가르쳐 '해당 분야의 전문가'로 만드는 과정이라고 볼 수 있습니다.

#### 💡 파인 튜닝이 필요한 순간

단순히 프롬프트를 잘 쓰는 것보다 파인 튜닝을 해야 할 때는 다음과 같습니다.

* **🗣️ 전문 용어/스타일 습득**: 업계 전용 은어, 사내 약어, 혹은 특정 브랜드의 어조(Tone & Manner)를 모델이 구사해야 할 때.
* **⚡ 비용 및 속도 최적화**: 매번 긴 프롬프트를 넣거나 거대 모델(GPT-4 등)을 쓰는 대신, 작은 모델을 튜닝하여 저비용·저지연(Latency)으로 운영하고 싶을 때.
* **🎯 정확도 향상**: 감성 분석, 의료 차트 분류 등 특정 태스크에서 범용 모델보다 더 높은 정확도가 필요할 때.
* **🧩 특이 사례(Edge Case) 처리**: 프롬프트만으로는 설명하기 힘든 복잡한 규칙이나 예외 상황을 모델이 이해해야 할 때.

#### 🔄 파인 튜닝 진행 4단계

1. **데이터 준비**: 고품질의 작업별 데이터셋을 수집하고 정제합니다. (데이터 품질이 가장 중요!)
2. **접근 방식 선택**:
   * **전체 파인 튜닝**: 모델의 모든 파라미터를 업데이트 (비용 높음).
   * **PEFT (Parameter-Efficient Fine-Tuning)**: 모델의 대부분을 고정하고 **핵심 파라미터만 추가 학습**하는 효율적 방식. (본 가이드에서 사용할 **LoRA**가 여기에 해당)
3. **모델 학습**: 준비된 데이터로 모델을 학습(Train)시키며 하이퍼파라미터를 조정합니다.
4. **평가 및 배포**: 학습된 모델이 새로운 데이터(Test set)에도 잘 동작하는지 검증하고 실서비스에 배포합니다.

> **⚠️ 주의사항 (Challenges)**
>
> * **과적합(Overfitting)**: 모델이 학습 데이터만 달달 외워서, 조금만 다른 질문에는 대답하지 못하는 현상입니다.
> * **치명적 망각(Catastrophic Forgetting)**: 새로운 전문 지식을 배우느라 기존의 일반 상식(기본 언어 능력 등)을 잊어버리는 현상입니다.

#### 🔄 파인 튜닝 진행 4단계

1. **데이터 준비**: 고품질의 작업별 데이터셋을 수집하고 정제합니다. (데이터 품질이 가장 중요!)
2. **접근 방식 선택**:
   * **전체 파인 튜닝**: 모델의 모든 파라미터를 업데이트 (비용 높음).
   * **PEFT (Parameter-Efficient Fine-Tuning)**: 모델의 대부분을 고정하고 **핵심 파라미터만 추가 학습**하는 효율적 방식. (본 가이드에서 사용할 **LoRA**가 여기에 해당)
3. **모델 학습**: 준비된 데이터로 모델을 학습(Train)시키며 하이퍼파라미터를 조정합니다.
4. **평가 및 배포**: 학습된 모델이 새로운 데이터(Test set)에도 잘 동작하는지 검증하고 실서비스에 배포합니다.

###

> **⚠️ 주의사항 (Challenges)**
>
> * **과적합(Overfitting)**: 모델이 학습 데이터만 달달 외워서, 조금만 다른 질문에는 대답하지 못하는 현상입니다.
> * **치명적 망각(Catastrophic Forgetting)**: 새로운 전문 지식을 배우느라 기존의 일반 상식(기본 언어 능력 등)을 잊어버리는 현상입니다.

### 1. 파인튜닝 환경 준비 (Setup)

가장 먼저 학습에 필요한 라이브러리를 설치합니다. Unsloth는 최적화된 커널을 사용하기 때문에 특정 버전의 의존성을 맞춰주어야 합니다.

#### 1.1 필수 라이브러리 설치

```
# Unsloth 및 최적화 라이브러리 설치
pip install unsloth
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

#### 1.2 추가 의존성 설치 (CMake/Curl)

Unsloth는 내부적으로 C++ 컴파일이 필요한 기능이나 GGUF 변환 등을 지원하기 위해 시스템 레벨의 도구가 필요합니다.

```
# Ubuntu/Colab 기준
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev cmake make -y
```

> **💡 Info** `xformers`는 메모리 효율적인 어텐션 연산을 지원하며, `trl`은 강화학습 및 SFT(Supervised Fine-Tuning)를 위한 라이브러리입니다.

### 2. 모델 로드 및 LoRA 구성

이제 베이스 모델을 불러오고, 효율적인 학습을 위해 \*\*LoRA(Low-Rank Adaptation)\*\*를 적용합니다. 앞서 설명한 **PEFT** 기법의 일종입니다.

#### 2.1 라이브러리 임포트 및 설정

```
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

# 모델 설정
# (예시: Qwen2.5-0.5B 등의 실제 모델명 사용 권장)
model_name = "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit" 
max_seq_length = 2048   # 입력 문맥 최대 길이
dtype = None            # None으로 설정 시 자동 감지 (Float16 등)
load_in_4bit = True     # 4bit 양자화로 메모리 절약
```

#### 2.2 모델 및 토크나이저 로드

`FastLanguageModel`을 사용하면 일반적인 HuggingFace `AutoModel`보다 훨씬 빠르게 모델을 로드할 수 있습니다.

```
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

#### 2.3 LoRA 어댑터 부착

전체 파라미터를 학습하는 대신, **LoRA** 어댑터를 붙여 일부 파라미터만 학습합니다. 이는 학습 속도를 비약적으로 높여줍니다.

```
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,             # LoRA Rank (높을수록 표현력 증가, 메모리 증가)
    target_modules = [  # 학습시킬 모듈 지정 (모든 선형 레이어 권장)
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,    # LoRA 스케일링 계수
    lora_dropout = 0,   # 0 권장 (최적화 문제 방지)
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 메모리 절약 기술
    random_state = 3407,
)
```

### 3. 데이터셋 준비

학습할 데이터를 불러와 모델이 이해할 수 있는 **Chat Template** 형태로 변환합니다.

#### 3.1 데이터 로드

```
# JSONL 파일에서 데이터 로드
dataset = load_dataset("json", data_files="6g_ai_dataset_augmented.jsonl", split="train")
```

#### 3.2 포맷 변환 (Formatting)

Qwen 모델은 대화형 모델이므로, `Instruction`(지시)과 `Output`(답변) 쌍을 대화 형식으로 만들어주어야 합니다.

```
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        # Qwen 채팅 템플릿 구조 정의
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        # 토크나이저를 통해 텍스트화 (EOS 토큰 등 자동 처리)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
        
    return {"text": texts}

# 데이터셋에 변환 적용
train_dataset = dataset.map(formatting_prompts_func, batched=True)

# 변환 결과 확인
print(train_dataset[0]["text"])
```

### 4. 학습(Training) 실행

준비된 모델과 데이터를 `SFTTrainer`에 연결하여 본격적인 학습을 시작합니다.

#### 4.1 Trainer 설정

```
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # 배치 사이즈
        gradient_accumulation_steps = 4,  # 그래디언트 누적 (실제 배치 2*4=8 효과)
        warmup_steps = 5,
        max_steps = 60,                   # 총 학습 스텝 (데이터 양에 따라 조절)
        learning_rate = 2e-4,             # 학습률
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),   # Ampere GPU 이상이면 bf16 사용
        logging_steps = 1,
        optim = "adamw_8bit",             # 8bit 옵티마이저로 메모리 절약
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

#### 4.2 학습 시작

```
trainer_stats = trainer.train()
```

> **🚀 Tip** `max_steps`는 예시입니다. 데이터셋이 크다면 `num_train_epochs`를 사용하거나 `max_steps`를 늘리세요.

### 5. 모델 테스트 (Inference)

학습이 끝난 모델이 질문에 올바르게 답변하는지 테스트합니다.

```
from transformers import TextStreamer

# 1. 추론 모드로 전환 (속도 최적화)
FastLanguageModel.for_inference(model)

# 2. 테스트 질문 작성
messages = [
    {"role": "user", "content": "6G 네트워크에서 AI의 역할은 무엇인가요?"}
]

# 3. 입력 토큰화
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# 4. 답변 생성 및 출력
_ = model.generate(
    input_ids=inputs,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
)
```

### 6. 모델 저장 (Export)

학습된 LoRA 어댑터와 설정을 저장합니다. GGUF 변환은 환경에 따라 불안정할 수 있으므로, 가장 호환성이 좋은 **HuggingFace 포맷** 저장을 권장합니다.

```
save_dir = "outputs/"

# 모델 및 토크나이저 저장
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ 모델 저장 완료! 저장 위치: {save_dir}")
```

### 7. 로드 및 검증 (Reload Test)

세션을 재시작했을 때, 저장된 모델을 정상적으로 불러올 수 있는지 검증하는 단계입니다.

```
from unsloth import FastLanguageModel

save_dir = "outputs/"

# 1. 저장된 경로에서 모델 로드
loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_dir, # 저장된 폴더 경로 지정
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 추론 모드 전환
FastLanguageModel.for_inference(loaded_model)

# 3. 재검증
messages = [{"role": "user", "content": "6G 네트워크에서 AI의 역할은 무엇인가요?"}]
inputs = loaded_tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

output = loaded_model.generate(input_ids=inputs, max_new_tokens=256)
print(loaded_tokenizer.decode(output[0], skip_special_tokens=True))
```

### 8. 요약 및 활용 (Summary)

이 가이드를 통해 우리는 다음 과정을 수행했습니다:

1. **Unsloth 환경 구축**: Qwen 0.6B/0.5B 모델을 4bit로 로드하여 메모리 효율 확보
2. **데이터셋 변환**: Raw JSONL 데이터를 LLM이 이해하는 Chat Format으로 변환
3. **LoRA 파인튜닝**: 적은 리소스로 모델의 지식을 특정 도메인(6G, AI 등)에 맞게 튜닝
4. **저장 및 검증**: 학습된 가중치를 저장하고 다시 불러와 추론 테스트

이 워크플로우는 **사내 구축형(On-Premise) LLM**, **로봇 제어용 경량 모델**, **모바일 온디바이스 AI** 등을 개발할 때 핵심적인 기초가 됩니다.
