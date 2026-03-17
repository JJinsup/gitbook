---
description: >-
  NVIDIA Isaac Lab은 NVIDIA Omniverse 플랫폼 위에서 작동하며, 복잡한 로봇 시뮬레이션과 고성능 학습 라이브러리를
  연결하는 가교 역할을 합니다. GPU 가속을 극대화하여 수천 개의 로봇을 동시에 시뮬레이션하고 학습시킬 수 있는 최첨단 프레임워크입니다.
icon: user-robot
---

# \[27] Isaac Lab: Tutorial

<figure><img src="../.gitbook/assets/image (6) (1).png" alt=""><figcaption></figcaption></figure>

{% embed url="https://github.com/isaac-sim/IsaacLab" %}

{% embed url="https://isaac-sim.github.io/IsaacLab/release/2.3.0/index.html" %}

{% embed url="https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html" %}

### 1. 개요

과거의 로봇 시뮬레이션이 단순히 동작을 확인하는 용도였다면, **Isaac Lab은 로봇의 '지능'을 학습시키는 데 초점**을 맞춥니다. 수천 개의 환경을 병렬로 구동하여 학습 시간을 획기적으로 단축하며, 실제 세계로의 전이(Sim-to-Real)를 고려한 정교한 물리 모델링을 제공합니다.

### 2. 에코시스템 구조

Isaac Lab은 크게 세 가지 계층으로 구성된 에코시스템을 가지고 있습니다.

#### ① 기초 계층: NVIDIA Isaac Sim & Omniverse

* **물리 엔진**: **PhysX**를 통한 고성능 GPU 가속 물리 연산.
* **렌더링**: **RTX 기술**을 활용한 실사 수준의 데이터 생성.
* **데이터 형식**: USD(Universal Scene Description)를 기반으로 한 유연한 asset 관리.

#### ② 핵심 계층: Isaac Lab Framework

* **자산 및 환경 관리**: 로봇, 센서, 물체를 프로그래밍 방식으로 쉽게 배치하고 구성.
* **프레임워크 독립성**: 특정 학습 라이브러리에 종속되지 않는 유연한 구조 설계.
* **병렬화 기술**: 시뮬레이션 데이터를 CPU 메모리 복사 없이 GPU 상에서 직접 학습 알고리즘으로 전달(**Zero-copy**).

#### ③ 상위 계층: 학습 라이브러리 (Learning Libraries)

Isaac Lab은 다음과 같은 주요 외부 강화 학습(RL) 라이브러리들과 호환됩니다.

| 라이브러리                       | 특징                                     |
| --------------------------- | -------------------------------------- |
| **RSL\_RL**                 | NVIDIA에서 주로 사용하는 대규모 병렬 학습에 최적화된 라이브러리 |
| **Stable Baselines3 (SB3)** | 가장 널리 사용되는 표준 강화 학습 프레임워크 (커뮤니티 지원 강함) |
| **skrl**                    | PyTorch 및 JAX 기반의 고속 학습을 지원하는 라이브러리    |
| **RL Games**                | 매우 높은 처리량을 자랑하는 고성능 강화 학습 라이브러리        |

### 3. 주요 특징 (Key Features)

* **GPU 가속 (GPU-Accelerated)**: 수천 개의 환경을 병렬로 구동하여 학습 시간을 며칠에서 몇 시간 단위로 단축합니다.
* **모듈화된 설계 (Modularity)**: 로봇 모델, 센서 설정, 보상 함수(Reward Function)를 모듈별로 관리하여 높은 재사용성을 제공합니다.
* **다양한 로봇 자산 지원**:
  * **사족 보행**: ANYmal, Unitree Go1/Go2 등
  * **협동 로봇**: Franka Panda, UR10 등
  * **모바일 로봇**: 각종 AMR 및 휠 기반 로봇
* **합성 데이터 생성 (SDG)**: AI 학습을 위한 고품질 이미지 및 세그멘테이션 데이터를 생성하는 **Replicator**와 긴밀하게 통합됩니다.

### 4. 왜 Isaac Lab인가? (Why Isaac Lab?)

1. **Sim-to-Real 간극 최소화**: 정교한 물리 모델과 노이즈 모델링을 통해 가상 세계에서 학습한 정책(Policy)을 실제 로봇에 바로 적용할 수 있도록 돕습니다.
2. **뛰어난 확장성**: 단일 GPU 워크스테이션부터 대규모 GPU 클러스터 환경까지 동일한 코드로 확장하여 실행할 수 있습니다.
3. **활발한 커뮤니티**: GitHub을 통해 전 세계 로봇 공학자들이 기여하고 있으며, 최신 강화 학습 알고리즘이 빠르게 반영됩니다.

> **Isaac Lab 학습의 핵심** Isaac Lab은 단순한 시뮬레이션 툴이 아니라, 로봇의 뇌(Brain)를 만들기 위한 거대한 실험실입니다. `Gym` 인터페이스와 유사한 환경 구성을 통해 기존 RL 연구자들이 쉽게 접근할 수 있도록 설계되어 있습니다.

### 5. Installing Isaac Lab with uv

`uv` 패키지 매니저를 사용하여 가상환경을 구축하고 Isaac Lab을 설치하는 현대적인 방식입니다.

#### 5.1 가상환경 생성 및 활성화

```
conda deactivate
cd /data2/[본인 계정]/isaac

# 가상환경 생성 (Python 3.11)
uv venv --python 3.11 env_isaaclab

# 가상환경 활성화
source env_isaaclab/bin/activate
```

#### 5.2 Isaac Sim Pip 패키지 설치

NVIDIA 전용 PyPI 인덱스를 통해 Isaac Sim 관련 패키지를 설치합니다.

```
uv pip install "isaacsim==5.1.0" "isaacsim-extscache==5.1.0" --extra-index-url https://pypi.nvidia.com
```

#### 5.3 Isaac Lab 설치

```
# Isaac Lab 클론
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 의존성 및 학습 프레임워크(rsl_rl, sb3 등) 한꺼번에 설치
./isaaclab.sh --install
```

#### 5.4 설치 확인

정상적으로 설치되었는지 튜토리얼 스크립트와 학습 스크립트를 실행해 봅니다.

```
# 빈 시뮬레이션 환경 실행 확인 (WebRTC 스트리밍 2번 모드)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2

# 개미 로봇(Ant) 강화학습 실행 확인 (Headless 모드)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

### 6. 🛠️ uv 기본 명령어 치트시트

`uv`를 사용할 때 자주 쓰이는 핵심 명령어 모음입니다.

| 분류         | 명령어                          | 설명                          |
| ---------- | ---------------------------- | --------------------------- |
| **환경 생성**  | `uv venv [이름] --python [버전]` | 특정 버전 파이썬 가상환경 생성           |
| **환경 활성화** | `source [이름]/bin/activate`   | 가상환경 활성화 (Linux/macOS)      |
| **환경 삭제**  | `rm -rf [이름]`                | uv는 별도 삭제 명령 없이 폴더만 삭제 가능   |
| **패키지 설치** | `uv pip install [패키지명]`      | 현재 가상환경에 패키지 설치             |
| **패키지 삭제** | `uv pip uninstall [패키지명]`    | 설치된 패키지 삭제                  |
| **목록 확인**  | `uv pip list`                | 현재 환경 설치 패키지 확인             |
| **동기화**    | `uv sync`                    | pyproject.toml 기반 의존성 자동 설치 |
| **캐시 삭제**  | `uv cache clean`             | 용량 확보를 위해 다운로드 캐시 삭제        |

> **성능 최적화**: `uv`는 설치 과정에서 하드 링크를 사용하기 때문에 디스크 공간을 절약하면서도 설치 속도가 매우 빠릅니다. Isaac Lab과 같이 거대한 의존성을 가진 프로젝트에서 매우 유리합니다.
