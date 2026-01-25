---
description: >-
  이 문서는 Isaac Lab 환경에서 SO-ARM100 / SO-ARM101 로봇 팔을 시뮬레이션하고 강화 학습을 수행하기 위한
  isaac_so_arm101 프로젝트 설정 및 사용 가이드를 다룹니다.
icon: user-robot
---

# \[28] Isaac Lab: So-arm-101

{% embed url="https://github.com/JJinsup/isaac_so_arm101" %}

### 0. 프로젝트 상세 구조 가이드

이 구조는 NVIDIA Isaac Lab의 External Extension 표준을 따르고 있으며, 로봇의 물리적 모델링(Assets)과 학습 로직(Tasks)이 체계적으로 분리되어 있습니다.

#### 0.1 디렉토리 트리 (src/isaac\_so\_arm101/)

```
src/isaac_so_arm101/
├── __init__.py                # Extension 및 태스크 등록 관문
├── robots/                    # [로봇 정의] 물리적 하드웨어 구성
│   ├── trs_so100/             # SO-ARM 100 관련 자산
│   ├── trs_so101/             # SO-ARM 101 관련 자산 (주력 모델)
│   │   ├── urdf/              # 로봇의 관절 및 링크 정보 (URDF 파일)
│   │   ├── __init__.py
│   │   ├── so_arm101.py       # 로봇 본체 설정 (강성, 마찰력, 파라미터)
│   │   └── LICENSE
│   └── __init__.py
├── tasks/                     # [태스크 설계] RL 학습 환경 및 시나리오
│   ├── lift/                  # 물체 들어 올리기 (Lift) 태스크
│   │   ├── agents/            # 학습 알고리즘 설정 (PPO 등)
│   │   │   ├── __init__.py
│   │   │   └── rsl_rl_ppo_cfg.py
│   │   ├── mdp/               # MDP(의사결정 과정) 핵심 로직
│   │   │   ├── __init__.py
│   │   │   ├── observations.py # 로봇이 보는 정보 (조인트, 위치 등)
│   │   │   ├── rewards.py      # 보상 체계 (학습 효율의 핵심)
│   │   │   └── terminations.py # 종료 조건 (리셋 타이밍)
│   │   ├── __init__.py
│   │   ├── joint_pos_env_cfg.py # 관절 제어 기반 환경 구성
│   │   └── lift_env_cfg.py    # Lift 태스크 최종 환경 조립
│   ├── reach/                 # 목표 지점 도달 (Reach) 태스크
│   │   ├── agents/
│   │   ├── mdp/
│   │   ├── __init__.py
│   │   ├── joint_pos_env_cfg.py
│   │   └── reach_env_cfg.py
│   └── __init__.py
└── scripts/                   # [실행 도구] 터미널 명령어 엔트리포인트
    ├── rsl_rl/                # RSL_RL 라이브러리 연동 스크립트
    │   ├── cli_args.py        # 명령줄 인자(num_envs 등) 정의
    │   ├── play.py            # 학습 모델 실행 (uv run play ...)
    │   └── train.py           # 학습 시작 (uv run train ...)
    ├── list_envs.py           # 사용 가능한 태스크 목록 확인
    ├── random_agent.py        # 무작위 행동 테스트용
    └── zero_agent.py          # 기본 자세(Zero pose) 테스트용
```

#### 0.2 주요 폴더별 상세 설명

1. **robots/** - **로봇 자산 정의 (Asset Definition)**
   * 로봇 팔 자체의 물리적 특성과 Isaac Sim 상에서의 구성을 담당합니다.
   * **so\_arm101.py**: URDF를 읽어와 Isaac Sim의 `ArticulatedRobot` 객체로 변환하고, 조인트 강성(Stiffness), 감쇠(Damping), 마찰력 등 물리 파라미터를 설정하는 핵심 파일입니다.
2. **tasks/** - **학습 환경 설계 (MDP & Environment)**
   * 로봇이 무엇을 배울지 결정하는 **"두뇌"** 설계도입니다.
   * **mdp/observations.py**: 로봇이 환경에서 관측하는 정보(조인트 각도, 물체 위치 등)를 정의합니다.
   * **mdp/rewards.py**: 학습의 성패를 가르는 보상 로직이 담겨 있습니다. "왜 로봇이 물체를 안 들지?"에 대한 답을 수정하는 곳입니다.
   * **agents/**: PPO 알고리즘의 하이퍼파라미터(학습률, 신경망 크기 등)를 조절합니다.
3. **scripts/** - **실행 및 테스트 (Entry Points)**
   * 사용자가 터미널에서 직접 실행하는 명령어들의 실제 구현체입니다.
   * `train.py`로 학습을 시작하고, `play.py`로 결과물을 확인하며, `zero_agent.py`로 초기 물리 모델을 검증합니다.

#### 0.3 구성 요소 간의 데이터 흐름 (Data Flow)

1. **scripts/train.py** 실행 시, tasks/의 환경 설정과 robots/의 물리 모델을 결합해 가상 세계를 생성합니다.
2. 로봇은 **mdp/observations.py**를 통해 세상을 인지하고 행동합니다.
3. 그 행동의 결과로 **mdp/rewards.py**에서 점수를 받아 agents/의 신경망을 업데이트합니다.

### 1. 프로젝트 클론 및 환경 설정

본 프로젝트는 `uv` 패키지 매니저를 사용하여 의존성을 관리합니다.

#### 1.1 uv 설치 및 프로젝트 클론

```bash
# 기존 콘다 환경 비활성화
conda deactivate

# uv 설치 (이미 설치된 경우 생략 가능)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 설치 후 터미널을 재시작하거나 설정을 반영하세요.

# 프로젝트 클론
cd /data2/[사용자계정]/isaac
git clone https://github.com/JJinsup/isaac_so_arm101
cd isaac_so_arm101

# 의존성 동기화 및 설치
uv sync
```

#### 1.2 가상환경 재활성화

터미널을 새로 열거나 프로젝트 폴더로 다시 돌아와 작업을 재개할 때는 다음 명령어로 가상환경을 활성화해야 합니다.

```shellscript
# 프로젝트 폴더 내에서 실행
source .venv/bin/activate
```

### 2. Quickstart

설치가 완료되면 다음 명령어를 통해 환경이 정상적으로 로드되는지 확인할 수 있습니다.

```shellscript
# 사용 가능한 학습 환경(Task) 리스트 확인
uv run list_envs

# Zero Agent 테스트 (동작 입력 없이 로봇 로드 확인)
uv run zero_agent --task SO-ARM100-Reach-Play-v0 --headless --livestream 2

# Random Agent 테스트 (무작위 동작 수행)
uv run random_agent --task SO-ARM100-Reach-Play-v0 --headless --livestream 2
```

### 3. 디버깅 및 동작 테스트 (Debugging)

#### 3.1 Zero Agent (로봇 불러오기 테스트)

```shellscript
uv run zero_agent --task Isaac-SO-ARM100-Reach-Play-v0 --headless --livestream 2
```

#### 3.2 Random Agent (관절 움직임 테스트)

```shellscript
uv run random_agent --task Isaac-SO-ARM100-Reach-v0 --headless --livestream 2
```

### 4. 강화 학습 (Reinforcement Learning)

<figure><img src="../.gitbook/assets/474230167-890e3a9d-5cbd-46a5-9317-37d0f2511684.gif" alt=""><figcaption></figcaption></figure>

#### 4.1 Reach Task (목표 도달)

* **학습(Training)**: \
  `uv run train --task Isaac-SO-ARM100-Reach-v0 --headless --num_envs 1024`
* **재생(Play)**: \
  `uv run play --task Isaac-SO-ARM100-Reach-Play-v0 --headless --livestream 2 --num_envs 16`

#### 4.2 Lift Task (물체 들어올리기)

* **학습(Training)**: \
  `uv run train --task Isaac-SO-ARM100-Lift-Cube-v0 --headless --num_envs 1024`
* **재생(Play)**: \
  `uv run play --task Isaac-SO-ARM100-Lift-Cube-Play-v0 --headless --livestream 2 --num_envs 16`

### 5. 트러블슈팅 (Troubleshooting)

**#**&#x31; **"No space left on device" (errno=28)**

```bash
# 감시 가능한 최대 파일 수 증가
sudo sysctl fs.inotify.max_user_watches=524288
# 설정 적용
sudo sysctl -p
```

> **Task Name 주의**: 환경 설정에 따라 태스크 이름 앞에 `Isaac-` 접두사가 붙거나 붙지 않을 수 있습니다. `uv run list_envs`로 출력되는 정확한 이름을 확인 후 사용하세요.
