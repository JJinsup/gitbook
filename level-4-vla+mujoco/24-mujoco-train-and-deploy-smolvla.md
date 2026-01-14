---
description: >-
  이 장의 목표는 Language → Policy → Action 구조가 실제로 어떻게 작동하는지, ACT에서 보았던 단순한 “행동 재현”을
  넘어 "언어 조건"에 따라 로봇의 행동이 지능적으로 달라지는 정책을 경험합니다.
icon: brain-circuit
---

# \[24] MuJoCo: Train & Deploy smolVLA

{% embed url="https://huggingface.co/docs/lerobot/smolvla" %}

### 24.1 왜 smolVLA인가?

<figure><img src="../.gitbook/assets/image (46).png" alt=""><figcaption></figcaption></figure>

기존의 VLA(Vision-Language-Action) 모델들은 파라미터 수가 너무 커서 학생 실습이나 개인 연구용으로 접근하기 매우 어려웠습니다. **smolVLA**는 이 문제를 해결하기 위해 설계된 경량화 모델입니다.

#### 실습 관점에서의 핵심 장점

* **접근성**: 소비자급 GPU(RTX 3090 등)에서도 원활한 학습과 추론이 가능합니다.
* **투명성**: 구조가 단순하여 시각 정보와 언어가 어떻게 행동으로 변환되는지 파악하기 좋습니다.
* **편의성**: Hugging Face 생태계를 기반으로 하여 데이터와 모델 관리가 쉽습니다.

{% hint style="info" %}
이 실습에서 smolVLA를 사용하는 이유는 단순한 성능 과시가 아니라, **최신 AI 로봇의 구조를 명확히 이해**하기 위함입니다.
{% endhint %}

### 24.2 smolVLA 구조 및 핵심 기술

#### 1) 구조 한 장 요약

smolVLA는 크게 두 개의 블록으로 구성됩니다.

> **\[Image + Language]** → **VLM Backbone** (인식) → **Action Expert** (행동 생성) → **Action** (출력)

<figure><img src="../.gitbook/assets/image (45).png" alt=""><figcaption></figcaption></figure>

#### 2) VLM Backbone (보는 부분)

* **SmolVLM-2 기반**: 사전 학습된 모델의 앞쪽 16개 레이어만 사용합니다.
* **레이어 축소 이유**: 로봇 제어에는 복잡한 추론보다 **빠르고 안정적인 인식**이 중요하기 때문입니다. 연산량을 줄여 추론 속도를 극대화했습니다.

#### 3) Action Expert (움직이는 부분)

* **Flow Matching**: Transformer 기반의 약 1억 파라미터 규모로, 연속적인 로봇 행동(Action Chunk)을 생성합니다.
* **언어 결합**: ACT와 달리 언어 명령이 행동 생성에 직접적인 영향을 미치도록 설계되었습니다.

#### 4) 효율성을 위한 아이디어

* **시각 토큰 최적화**: Pixel Shuffle 기술을 통해 프레임당 토큰을 64개로 제한하여 지연 시간(Latency)을 줄였습니다.
* **Interleaved Attention**: 언어/시각 정보를 참조하는 Cross-Attention과 이전 행동의 흐름을 유지하는 Self-Attention을 번갈아 배치하여 **언어 지시에 따르면서도 부드러운 동작**을 유지합니다.
* **비동기 추론 (Asynchronous Inference)**: 행동 실행과 다음 예측을 분리하여 로봇이 생각하느라 멈추는 현상을 방지했습니다.

### 24.3 ACT와 smolVLA의 차이

| 항목        | ACT                       | smolVLA                           |
| --------- | ------------------------- | --------------------------------- |
| **핵심 입력** | 시각 정보(Vision) + 상태(State) | Vision + State + **언어(Language)** |
| **판단 방식** | 규정된 동작의 단순 재현             | 언어 조건에 따른 상황 판단                   |
| **실습 목표** | 파이프라인 완주 및 재현             | **언어 기반 의미 이해(근사)**               |

> **차이점 이해하기**
>
> * **ACT**: “이 상태에서는 이 행동을 했었다”를 기억해서 복사함
> * **smolVLA**: “이 말을 들었을 때, 지금 상황에서 무엇을 해야 하는가?”를 조건부로 판단함

### 24.4 로컬: Language 데이터 생성

언어 명령어가 포함된 데모 데이터를 수집합니다.

```
# Language 데이터 수집 스크립트 실행
python 5.language_env_standalone.py
```

<figure><img src="../.gitbook/assets/image (47).png" alt=""><figcaption></figcaption></figure>

#### 🛠️ 데이터 수집 흐름

1. **설정값 지정**: `NUM_DEMO`(수집 개수), `ROOT`(저장 폴더), `xml_path`(언어 환경 씬) 등을 설정합니다.
2. **데이터셋 생성**: 기존 폴더가 있다면 삭제 여부를 묻고, `LeRobotDataset.create()`를 통해 VLA 학습에 필요한 Feature 스펙을 정의합니다.
3. **언어 환경(`SimpleEnv2`) 실행**: `SimpleEnv2`는 내부에 현재 수행해야 할 지시문(PnPEnv.instruction)을 포함하고 있습니다.
4. **기록 시작(`record_flag`)**: 로봇의 첫 움직임(Action != 0)이 감지되면 기록을 시작하며 콘솔에 현재 Task를 출력합니다.
5. **프레임 저장 (20Hz)**: 이미지, 상태, 초기 물체 위치(`obj_init`)와 함께 **가장 중요한`task(=PnPEnv.instruction)`** 정보를 매 타임스텝 저장합니다.
6. **성공 및 리셋**: `check_success()`가 True가 되면 에피소드를 저장하고 환경을 리셋하여 다음 지시문으로 넘어갑니다.

#### 📌 데이터 수집 핵심 포인트

1. **결과물**: `demo_data_language/` 폴더가 생성됩니다.
2. **데이터 구성**: 각 에피소드에는 로봇의 동작 데이터와 함께 자연어 지시문(Instruction)이 함께 저장됩니다.
3. **학습 데이터의 질**: smolVLA는 구조가 효율적이므로, "많은 데이터"보다 "잘 설계된 데이터(정확한 언어-동작 매칭)"가 훨씬 중요합니다.
4. **이미지 데이터 보존**: 이전 ACT 코드와 달리 `images/` 폴더를 함부로 지우지 않습니다. 이는 데이터셋 로드 시 경로 참조 오류를 방지하기 위함입니다.

#### 🆚 ACT vs Language 수집 코드 차이점

| 구분          | ACT (SimpleEnv)       | Language (SimpleEnv2)      |
| ----------- | --------------------- | -------------------------- |
| **환경**      | 단일 태스크, 언어 지시 없음      | 에피소드마다 다른 언어 지시문 포함        |
| **Task 필드** | 고정된 문자열 (`TASK_NAME`) | 매 에피소드마다 변하는 `instruction` |
| **데이터 의미**  | "이 상태에서 사람이 한 행동"     | "이 말을 들었을 때 사람이 한 행동"      |
| **물체 정보**   | 기본 물체 초기 상태           | 다수 물체 상황을 포함한 확장 상태        |
| **학습 결과**   | 특정 동작의 단순 재현          | 언어 조건에 따른 지능적 행동 선택        |

### 24.5 로컬: 데이터 시각화 및 관리 (HF Bridge)

수집한 데이터를 시각화하여 확인하거나, 서버 학습을 위해 Hugging Face로 업로드/다운로드하는 통합 관리 스크립트입니다.

```
python 6.visualize_data_standalone.py
```

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 22시 37분 25초.gif" alt=""><figcaption></figcaption></figure>

* **Rollout**: 로컬 데이터가 제대로 수집되었는지 애니메이션으로 확인합니다.
* **Push to Hub**: 수집된 데이터를 **Hugging Face**에 업로드하여 서버와 로컬을 잇는 가교 역할을 수행합니다.

#### 1) 설정 파일(YAML) 상세 가이드

이 스크립트는 `smolvla_omy.yaml` 파일을 읽어 동작합니다. 각 항목의 의미를 이해하는 것이 중요합니다.

**① Dataset: 데이터 위치 및 식별자**

```
dataset:
  repo_id: "JJinsup/omy_pnp_language" # HF Hub 데이터셋 식별자
  root: "./demo_data_language"       # 로컬 데이터 저장 경로
```

* **repo\_id**: `유저명/데이터셋명` 형식. 다운로드 및 업로드 기능의 기준이 됩니다.
* **root**: 로컬에서 `data/`, `meta/` 등이 실제로 존재하는 경로입니다.
* **CHECK** : `./demo_data_language` 안에 실제 데이터가 있는지, `repo_id`가 본인의 계정명으로 되어 있는지 확인하세요.

**② Policy: 학습 및 출력 방식**

```
policy:
  type: smolvla
  chunk_size: 5
  n_action_steps: 5
  device: cuda
```

* **chunk\_size**: 모델이 한 번에 예측하는 행동 묶음의 길이입니다.
* **n\_action\_steps**: 실제로 환경에 적용할 스텝 수입니다. 보통 `chunk_size`와 일치시킵니다.
* **TIP** : 청크를 키우면 움직임이 부드러워지지만 학습 난이도가 올라갑니다. 실습 시에는 5 \~10이 적당합니다.

**③ Output & Checkpoint: 저장 위치**

```
save_checkpoint: true
output_dir: ./ckpt/smolvla_omy
```

* **output\_dir**: 학습 결과물(`ckpt`)이 저장되는 곳입니다. 추론(`deploy`) 시 이 경로를 참조합니다.
* **save\_checkpoint**: 학습 중간 과정을 저장할지 여부를 결정합니다.

**④ 학습 실행 세팅 (Execution)**

```
batch_size: 32
num_workers: 8
steps: 20_000
seed: 42
```

* **batch\_size**: GPU 메모리(VRAM)에 따라 조절합니다. 메모리 부족 시 32 -> 16 -> 8로 줄이세요.
* **steps**: 총 업데이트 횟수입니다. 테스트 시에는 2,000 \~ 5,000, 본 학습 시에는 20,000 이상을 권장합니다.

**⑤ 로그 및 저장 주기 (Frequency)**

* **log\_freq (50)**: 50 스텝마다 학습 로그(Loss 등)를 출력합니다.
* **save\_freq (10,000)**: 10,000 스텝마다 모델을 저장합니다.
* **eval\_freq (-1)**: 평가 루프를 비활성화하여 실습 절차를 간소화합니다.

**⑥ Resume & Preset**

* **resume (false)**: 이전 학습 지점부터 이어서 할지 결정합니다.
* **use\_policy\_training\_preset (true)**: smolVLA를 위한 권장 최적 설정을 자동 적용합니다. (초보자 권장)

**⑦ WandB: 실험 기록**

```
wandb:
  enable: true
  project: smolvla_omy
  entity: "본인-wandb-ID"
  disable_artifact: true
```

* **entity**: 본인의 WandB ID 또는 팀 이름을 입력해야 로그가 정상 업로드됩니다.
* **disable\_artifact**: 불필요한 대용량 파일 업로드를 방지하여 트래픽을 아낍니다.

#### 2) 실행 모드 (핵심 옵션)

실행 시 붙이는 인자(`--`)에 따라 모드가 바뀝니다.

| 모드           | 실행 명령어                                  | 설명                              |
| ------------ | --------------------------------------- | ------------------------------- |
| **시각화 (기본)** | `python 6.visualize_data_standalone.py` | 로컬 데이터를 MuJoCo에서 에피소드 단위 재생     |
| **특정 시작점**   | `... --start_episode 5`                 | 5번 에피소드부터 재생 시작                 |
| **업로드**      | `... --push_to_hub`                     | 시각화 없이 즉시 Hugging Face로 데이터 업로드 |
| **다운로드**     | `... --download`                        | Hugging Face에서 데이터를 로컬로 내려받기    |

#### 3) 동작 원리 상세

* **업로드 모드**: `--push_to_hub` 옵션이 켜지면 MuJoCo 뷰어를 띄우지 않고 업로드 로직만 수행한 뒤 즉시 종료됩니다. (빠른 배포 가능)
* **시각화 모드**: `LeRobotDataset`을 로드한 뒤, `SimpleEnv2` 환경에 저장된 `obj_init`(초기 위치)과 `task`(지시문)를 복원하여 사람이 조작했던 데모를 그대로 재현합니다.

{% hint style="info" %}
**자주 헷갈리는 포인트 3개**

* **repo\_id**: "내 컴퓨터의 폴더 이름"이 아니라 **Hugging Face에 생성한 데이터셋 식별자**(`username/dataset_name`)여야 합니다.
* **업로드 확인**: `--push_to_hub`를 사용하기 전 반드시 `huggingface-cli login`이 되어 있어야 합니다.
* **경로 에러**: YAML 파일의 `root` 경로가 실제 데이터 폴더 위치와 다르면 로드 실패 에러가 발생합니다. 가장 흔한 에러이니 경로를 꼭 확인하세요.&#x20;
{% endhint %}

### 24.6 서버: smolVLA 학습 (GPU)

Hugging Face에 업로드된 언어 데이터를 불러와 학습을 진행합니다.

* **실행 파일**: `8.smolvla_train.ipynb`
* **핵심 지표**: `wandb`를 통해 학습 Loss와 이미지-텍스트 정렬 상태를 모니터링합니다.

#### 24.6.1 코드 설명

**1️⃣ 필수 라이브러리 설치**

```bash
!pip install num2words
!pip install accelerate
!pip install safetensors>=0.4.3
!pip install pytest
!pip install transformers==4.50.3
!pip install wandb
!pip install python-dotenv
```

* smolVLA는 **Transformers + Accelerate 기반 학습 코드**를 사용
* `wandb`는 실험 로그 기록
* `python-dotenv`는 API 키를 코드에 직접 쓰지 않기 위한 도구
* `transformers==4.50.3` **버전 고정 중요**
* 버전이 다르면 모델 로딩/학습 중 에러 발생 가능
* 서버에 이미 설치돼 있으면 다시 깔 필요 없음

***

**2️⃣ (선택) Hugging Face에서 데이터셋 다운로드**

```bash
!git clone https://huggingface.co/datasets/JJinsup/omy_pnp_language demo_data_language
```

* 로컬에서 수집한 데이터를 서버로 옮기지 않아도 됨
* Hugging Face를 **데이터 저장소**처럼 사용
* `smolvla_omy.yaml`의 `dataset.root`가 `./demo_data_language`로 되어 있어야 함
* 이미 데이터가 서버에 있으면 이 단계는 생략 가능

***

**3️⃣ wandb 로그인 (환경변수 방식)**

```python
import os
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb.login()
```

* `.env` 파일에 저장된 `WANDB_API_KEY`를 자동으로 불러옴
* 코드에 API 키를 직접 쓰지 않기 위한 **보안 방식**
*   `.env` 파일에 아래가 있어야 함

    ```
    WANDB_API_KEY=xxxxxxxxxxxxxxxx
    ```
* 한 번 로그인하면 같은 세션에서는 다시 안 해도 됨
* wandb 안 쓰려면 YAML에서 `wandb.enable: false`

***

**4️⃣ 이전 체크포인트 삭제 (깨끗한 재학습)**

```bash
!rm -rf ./ckpt/smolvla_omy
```

* 이전 실험 결과가 남아 있으면
  * resume 오동작
  * 잘못된 ckpt 로딩
* 진짜 **처음부터 학습**하려는 경우에만 실행
* 이어서 학습하려면 지우지 말고 YAML에서 `resume: true`

***

**5️⃣ smolVLA 학습 실행 (실습에서 가장 중요한 한 줄)**

```bash
!python train_model.py --config_path smolvla_omy.yaml
```

* `smolvla_omy.yaml` 설정을 읽어서:
  * 데이터셋 로드
  * smolVLA 정책 생성
  * wandb 로깅
  * 체크포인트 저장을 자동으로 수행
* 학습 로직은 안 건드리고, 설정(YAML)만 바꿔가며 실험하기 위함
* 학습 시작 시 콘솔에:
  * dataset 로딩 로그
  * wandb run 생성
  * loss 출력이 보이면 정상
* 처음엔 `steps`를 2,000 정도로 줄여서 “돌아가는지”만 확인하는 게 좋음

### 24.7 로컬: smolVLA 추론 및 검증

학습된 체크포인트를 불러와 MuJoCo 환경에서 로봇이 지시어에 따라 움직이는지 확인하는 최종 단계입니다.&#x20;

```
# smolVLA 추론 실행
python 8.deploy_smolvla_omy.py
```

`Loading policy...`\
`WARNING:root:Device 'None' is not available. Switching to 'cuda'.`\
`Reducing the number of VLM layers to 16 ...`\
`Loading weights from local directory`\
`Creating environment...`

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 22시 55분 35초 (online-video-cutter.com).gif" alt=""><figcaption></figcaption></figure>

<p align="center"><strong>추론 영상(4배속)</strong></p>

#### 1) 모델 로딩 및 준비 (`load_policy`)

```
policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
policy.eval()
```

* **동작**: `./ckpt/smolvla_omy`에 저장된 가중치와 데이터 통계량(stats)을 함께 불러옵니다.
* **CHECK** : 학습 시 사용한 데이터셋(`omy_pnp_language`)이 로컬에 존재해야 정규화 기준을 맞춰 정상 추론이 가능합니다.

#### 2) VLA 입력 데이터 구성 (핵심)

smolVLA가 판단을 내리기 위해 매 순간 입력받는 데이터 구조입니다.

```
data = {
    'observation.state': state,        # 로봇 관절 상태
    'observation.image': agent_image,  # 메인 카메라
    'observation.wrist_image': wrist, # 손목 카메라
    'task': [env.instruction]          # 현재 수행할 자연어 지시문
}
```

* **VLA의 정수**: 시각(V), 상태(S)뿐만 아니라 Language가 `task` 필드를 통해 직접 입력됩니다.
* **ACT와의 차이**: ACT는 고정된 동작을 재현하지만, smolVLA는 이 `task` 문장을 조건으로 사용하여 행동을 결정합니다.

#### 3) 추론 및 롤아웃 루프

1. **수집**: 현재 환경의 이미지, 상태, 지시어를 수집합니다.
2. **예측**: `policy.select_action(data)`가 다음 행동 묶음(Action Chunk)을 예측합니다.
3. **실행**: 예측된 액션을 MuJoCo 환경(`env.step`)에 적용합니다.

#### 4) 성공 판정 및 리셋

```
if env.check_success():
    policy.reset() # 액션 큐 초기화
    env.reset()    # 환경 초기화
```

* **연속 행동**: 한 가지 태스크를 성공하면 정책과 환경을 초기화하고 다음 지시문(Rollout)으로 자동으로 넘어갑니다. 로봇이 멈추지 않고 계속해서 새로운 명령을 수행하는지 확인하세요.

#### 🔍 관찰 포인트

1. **상황 대응**: 물체의 초기 위치를 수동으로 조금씩 옮겨보아도 모델이  목표를 수행하는가?
2. **언어 반응**: 입력 문장을 바꾸었을 때(예: 'Pick up' vs 'Push') 로봇의 행동이 변화하는가?

### 🏁 마무리: 우리가 경험한 파이프라인

1. **MuJoCo**: 물리 엔진과 시뮬레이션 환경 구축
2. **Offline Data**: 고품질 데이터셋 생성 기술
3. **Server Training**: 대규모 모델의 효율적 학습 전략
4. **VLA Structure**: 시각-언어-행동이 통합된 지능형 로봇 제어 체계

### 🧭 전체 흐름 한 줄 요약

> **\[22] 구조 이해 → \[23] ACT 파이프라인 완주 → \[24] 언어 기반 지능형 VLA 확장**

