---
description: >-
  이 장의 목표는 데이터 수집 → 서버 학습 → 로컬 추론이라는 전체 파이프라인을 직접 끝까지 완주, 이 실습을 통해 “정책 학습(Policy
  Learning)이 실제로 무엇을 의미하는지” 체감하는 것이 핵심입니다.
icon: brain-circuit
---

# \[23] MuJoCo: Train & Deploy ACT

{% embed url="https://huggingface.co/docs/lerobot/act" %}

<figure><img src="../.gitbook/assets/image (44).png" alt=""><figcaption></figcaption></figure>

### 23.1 실습 목표와 범위

| 구분      | 이번 장에서 수행하는 것               | 이번 장에서 하지 않는 것         |
| ------- | --------------------------- | ---------------------- |
| **환경**  | MuJoCo 시뮬레이터 내 로봇 조작        | 고성능 모델 하이퍼파라미터 튜닝      |
| **데이터** | Offline 데이터 수집 (성공 에피소드 저장) | 실시간 언어(Language) 입력 사용 |
| **학습**  | 서버 GPU를 이용한 ACT 정책 학습       | -                      |
| **배포**  | 학습된 모델을 로컬 환경에서 추론          | -                      |

{% hint style="info" %}
**중요**: 우리의 목표는 모델의 성능을 극대화하는 것이 아니라, **파이프라인의 전체 흐름을 완주**하는 것입니다.
{% endhint %}

### 23.2 로컬: 데이터 수집

#### 23.2.1 실행 방식 이해

데이터 수집 스크립트는 `NUM_DEMO` 값만큼 에피소드를 자동으로 연속 수집합니다. 즉, 한 번 실행하면 설정된 횟수만큼 성공할 때까지 데이터가 누적 저장됩니다.

```
# 데이터 수집 스크립트 실행
python 1.collect_data_standalone.py
```

<figure><img src="../.gitbook/assets/image (42).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/image (43).png" alt=""><figcaption></figcaption></figure>

<p align="center"><strong>생성된 데이터</strong></p>

#### 23.2.2 코드 설명 (`1.collect_data_standalone.py`)

**1) 주요 설정 (Configuration)**

가장 먼저 확인하고 수정해야 할 부분입니다.

* `TASK_NAME`: 데이터셋에 붙는 라벨 (예: 'Put mug cup on the plate')
* `NUM_DEMO`: **“몇 판 성공할 때까지 모을 거냐”** (최소 30\~50 권장)
* `ROOT`: 데이터가 물리적으로 저장될 경로 (`./demo_data`)
* `xml_path`: 어떤 시각적 환경(MuJoCo Scene)에서 실습할지 지정

**2) 데이터 관리 로직**

* `ROOT` 폴더가 이미 존재하면 삭제할지(`y`), 기존 데이터에 이어붙일지(`n`)를 묻습니다.
* **처음 실습**: `y`로 지우고 깨끗하게 시작하세요.
* **데이터 보강**: `n`으로 설정하여 에피소드를 추가 수집하세요.

**3) 수집 메커니즘 (Mapping)**

* **Observation**: 로봇이 본 것(카메라 이미지)과 현재 상태(관절 위치)
* **Action**: 사람이 조작한 행동(키보드 입력에 따른 관절 움직임)
* **Task**: 이 데이터가 도달하고자 하는 목표 문장

{% hint style="info" %}
**데이터 수집 팁**

* **Z 키**: 실수했을 때 환경을 초기화(Reset)합니다.
* **Record Flag**: 로봇이 가만히 있을 때는 저장하지 않다가, 실제 움직임이 감지되면 자동으로 기록을 시작합니다.
{% endhint %}

### 23.3 로컬: 데이터 시각화

수집한 데이터가 모델의 "재료"로서 적합한지 직접 확인합니다.

```
python 2.visualize_data_standalone.py
```

* **확인 사항**: 로봇의 움직임이 부드러운가? 카메라 이미지(Agent/Wrist)가 잘 찍혔는가?
* **관찰 포인트**: 타임스텝별로 변화하는 `state`와 `action`의 관계를 확인하며 “데이터가 정책의 재료”라는 감각을 익히세요.

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 21시 56분 21초.gif" alt=""><figcaption></figcaption></figure>

### 23.4 서버: 모델 학습

데이터가 준비되었다면, 이제 GPU 자원을 활용해 정책을 학습시킬 차례입니다.

> _**\[22]를 참고해서 서버에도 대표 학생 한명이 환경 세팅을 해주세요**_

#### 23.4.1 데이터 업로드

로컬에서 생성된 `demo_data/` 폴더를 서버의 `lerobot-mujoco-tutorial` 프로젝트 경로 안으로 업로드

#### 23.4.2 학습 실행

서버에서 아래 노트북 파일을 실행합니다.

* **파일명**: `3.train.ipynb`
* **내용**: ACT(Action Chunking Transformer) 정책 학습
* **확인**: 학습 완료 후 `ckpt/` 폴더 내에 모델 가중치 파일이 생성되었는지 확인합니다.

#### 23.4.3 코드 이해

**1️⃣ 데이터셋 메타데이터 로드 & Feature 분리**

```python
dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {k: ft for k, ft in features.items() if k not in output_features}
input_features.pop("observation.wrist_image")
```

* 정책은 **입력(observation)** → **출력(action)** 구조로 학습된다.
* 따라서 데이터셋의 feature를
  * 입력(feature)
  * 출력(label)으로 명확히 나눠야 한다.
* `wrist_image`는 학습을 단순화하기 위해 제외했다.

_CHECK_&#x20;

* `output_features`에 `action`만 들어 있는지
* `input_features`에서 image, state 등이 정상적으로 남아 있는지

_TIP_

* wrist 카메라도 쓰고 싶다면\
  `input_features.pop(...)` 줄을 제거하고 **데이터 수집 단계 feature와 반드시 일치시켜야 한다.**

***

**2️⃣ ACT 정책 설정 (Action Chunking)**

```python
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=10,
    n_action_steps=10
)

delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)
policy.train()
```

* ACT는 한 스텝의 action이 아니라 “action 묶음(chunk)”을 예측한다.
* `chunk_size=10` → 현재 상태에서 **앞으로 10스텝의 행동**을 한 번에 학습
* `chunk_size`와 `n_action_steps` 값이 동일한지
* deploy 단계에서도 동일한 설정을 쓰는지

_TIP_

* chunk 크기를 키우면
  * smoother한 행동
  * 하지만 학습은 더 어려워질 수 있다

***

**3️⃣ 데이터 증강: 이미지에 노이즈 추가**

```python
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])
```

* 실제 추론 환경은 데모 데이터와 완전히 같지 않다.
* 입력 이미지에 약간의 노이즈를 주면 **policy가 작은 시각적 변화에 덜 민감해진다.**

_CHECK_&#x20;

* 이미지 값이 `[0, 1]` 범위를 벗어나지 않는지
* 학습 초반 loss가 비정상적으로 커지지 않는지

_TIP_

* 학습이 불안정하면 `std=0.01`로 줄여도 된다.
* 처음 실습에서는 **노이즈 제거 후 비교 실험**도 추천

***

**4️⃣ Dataset & DataLoader 생성**

```python
dataset = LeRobotDataset(
    "omy_pnp",
    delta_timestamps=delta_timestamps,
    root='./demo_data',
    image_transforms=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True,
)
```

* Offline 학습이므로 **환경을 돌리지 않고 데이터만 반복적으로 읽는다.**
* `shuffle=True`는 episode 순서 편향을 줄이기 위함이다.

_CHECK_&#x20;

* `demo_data/` 경로가 올바른지
* `batch_size`가 GPU 메모리에 맞는지

_TIP_

* CUDA OOM(Out of Memory) 나면
  * `batch_size`부터 줄이기
  * 다음으로 `num_workers` 조정

***

**5️⃣ 오프라인 학습 루프**

```python
training_steps = 3000
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

step = 0
while step < training_steps:
    for batch in dataloader:
        inp_batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        loss, _ = policy.forward(inp_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.3f}")

        step += 1
        if step >= training_steps:
            break
```

* 이 학습은 강화학습이 아니라 **Behavior Cloning** 에 가깝다.
* “사람이 한 action을 그대로 맞추도록” 학습한다.

_CHECK_

* loss가 전반적으로 감소하는지
* NaN이나 갑자기 폭증하지 않는지

_TIP_

* 3000 step은 “돌아는 가는 수준”
* 그래프가 엉키면 5000\~10000 step 권장

***

### 6️⃣ 체크포인트 저장

```python
policy.save_pretrained('./ckpt/act_y')
```

* 학습과 추론을 분리하기 위함
* deploy 단계에서 이 ckpt를 불러 사용한다.

_CHECK_

* `ckpt/act_y/` 폴더 생성 여부
* 파일 크기가 비정상적으로 작지 않은지

***

**7️⃣ 간단 평가: pred vs GT action 비교**

```python
policy.eval()
policy.reset()

action = policy.select_action(inp_batch)
gt_action = inp_batch["action"][:, 0, :]
```

* 학습이 “대충이라도 되었는지” 빠르게 확인
* 정량 지표 + 시각적 비교

_CHECK_

* pred와 GT가 완전히 무관하지 않은지
* action 차원별로 편차가 큰 곳은 없는지

_TIP_

* 이 평가는 **정식 성능 평가가 아님**
* deploy에서 실제 움직임을 꼭 확인해야 한다

### 23.5 로컬: 모델 배포 및 추론

서버에서 학습된 `ckpt` 파일을 로컬로 다운로드한 후, 로봇이 스스로 움직이는지 확인합니다.

```
# 추론 스크립트 실행
python 4.deploy_standalone.py
```

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 22시 03분 25초.gif" alt=""><figcaption></figcaption></figure>

#### 23.5.1 코드 설명 (`4.deploy_standalone.py`)

**1) 핵심 설정 및 경로**

* `POLICY_PATH`: 학습된 모델 가중치가 저장된 경로 (`./ckpt/act_y`)
* `DATASET_ROOT`: 모델 학습 시 사용했던 데이터셋의 통계량(stats)을 참조하기 위해 동일한 데이터 경로가 필요합니다.

**2) Feature 구성 일치 (중요)**

학습 단계에서 `wrist_image`를 제외했다면, 배포 단계에서도 반드시 동일하게 제외해야 합니다.

```
input_features.pop("observation.wrist_image", None)
```

* **Tip**: 학습/배포 시 Feature 구성이 다르면 모델이 로드되더라도 로봇이 엉뚱하게 동작하거나 오류가 발생

**3) 정책 로드 및 Temporal Ensembling**

```
cfg = ACTConfig(..., temporal_ensemble_coeff=0.9)
policy = ACTPolicy.from_pretrained(POLICY_PATH, config=cfg, dataset_stats=dataset_metadata.stats)
```

* **Temporal Ensembling**: 이전 스텝에서 예측한 미래 행동과 현재 예측한 행동을 적절히 섞어(0.9 비중) 로봇의 움직임을 훨씬 더 부드럽게 만들어줍니다.

**4) 환경 초기화 및 물체 위치 맞추기**

ACT와 같은 BC(Behavior Cloning) 모델은 학습 데이터의 분포에 매우 민감합니다.

```
obj_init_pose = dataset[0]['obj_init'].numpy() # 학습 데이터의 물체 위치 로드
```

* **성공률 상승 비법**: 학습 데이터에 기록된 물체의 초기 위치를 읽어와 시뮬레이션 환경을 초기화하면, 모델이 학습한 상황과 동일해져 성공률이 비약적으로 상승합니다.

**5) 롤아웃 루프 (Observation → Policy → Action)**

매 타임스텝(20Hz)마다 아래 과정을 반복합니다:

1. **관측**: 현재 로봇 상태와 카메라 이미지를 가져옵니다.
2. **예측**: `policy.select_action(data)`를 통해 다음 행동을 결정합니다.
3. **실행**: `PnPEnv.step(action)`으로 로봇을 움직입니다.

{% hint style="info" %}
**반드시 짚고 넘어가야 할 포인트**

이 추론 결과는 고도의 '지능'이 아닙니다. 이것은 사람이 만든 데이터를 흉내 낸 결과일 뿐입니다.

**추론 문제 해결 (Troubleshooting)**

* **로봇이 멈춰있나요?**: `ckpt` 경로가 맞는지, 데이터셋 `stats`가 학습 시와 동일한지 확인하세요.
* **동작이 너무 거친가요?**: `temporal_ensemble_coeff` 값을 조정하거나 학습 횟수를 늘려야 합니다.
{% endhint %}

* 새로운 상황(물체 위치 변경 등)에는 매우 취약합니다.
* 조건이 조금만 바뀌어도 쉽게 실패합니다.
* **이 한계를 이해하는 것**이 다음 장인 \[24] smolVLA 실습(언어 조건부 VLA)으로 넘어가는 이유입니다.

### 🏁 한 줄 요약

> **ACT 실습은 “demo\_data의 통계량을 기반으로 정책을 로드하고, 실시간 이미지/상태 입력을 행동으로 변환하여 Pick & Place를 완주하는 과정”입니다.**
