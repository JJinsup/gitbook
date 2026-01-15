---
description: >-
  이 장의 목표는 실습을 바로 시작하는 것이 아닙니다. 이 저장소에서 우리가 무엇을 하고, 어디까지 할 것인지의 범위를 명확히 이해하는 것이
  목적입니다.
icon: brain-circuit
---

# \[22] MuJoCo:  lerobot-mujoco-tutorial

<figure><img src="../.gitbook/assets/image (39).png" alt=""><figcaption></figcaption></figure>



### 🎯 목표

이 장을 끝내면, 이후 실습(\[23], \[24])에서 “지금 내가 파이프라인의 어디에 있는지”를 항상 스스로 인식할 수 있어야 합니다.

### 22.1 실습 시나리오 한 장 요약

전체 워크플로우는 **로컬**과 **서버** 역할을 명확히 분리하여 진행됩니다.

#### 🔄 전체 흐름

1. **\[Local]** MuJoCo 시뮬레이터 실행 → 로봇 조작 → **데이터 수집**
2. **\[Server]** 로컬에서수집된 데이터를 서버로 전송 → **모델 학습 (GPU)**
3. **\[Local]** 학습된 모델 가중치 다운로드 → **추론 및 동작 확인**

#### 💻 역할 분리

* **Local (PC/Laptop)**: MuJoCo 환경 이해, 데이터 생성, 학습된 모델의 최종 추론 확인
* **Server (RTX 3090)**: GPU 기반의 고성능 모델 학습 전담

{% hint style="info" %}
**핵심 키워드**

* **Offline Dataset**: 실시간 학습이 아닌, 미리 수집된 데이터를 사용함
* **Policy Learning**: 수집된 데이터를 보고 로봇의 행동 규칙(정책)을 학습함
* **Deployment**: 학습된 모델을 실제 환경(시뮬레이터)에 적용함
{% endhint %}

### 22.2 프로젝트 폴더 구조 이해

{% embed url="https://github.com/JJinsup/lerobot-mujoco-tutorial" %}

이 저장소는 코드 자체보다 **구조를 이해하는 것이 실습의 절반**입니다. \
실습을 진행하며 다음 관점으로 폴더를 구분하세요.

1. **Original**: `git clone` 시 이미 존재하는 기본 코드 및 환경 설정
2. **Generated**: 실습(데이터 수집, 학습)을 진행하며 새로 생성되는 결과물 (`outputs/`, `data/` 등)
3. **Config**: 학생이 직접 수정해야 할 설정 파일 (`.yaml` 등)

```bash
lerobot-mujoco-tutorial/
├── 1.collect_data_standalone.py      # (실습) 로봇 텔레옵으로 데이터 수집
│
├── 2.visualize_data_standalone.py    # (실습) 수집한 데이터 시각화
│
├── 3.train.ipynb                     # (실습) 정책 모델 학습
│
├── 4.deploy_standalone.py            # (실습) 학습된 모델로 시뮬레이션 실행
│
├── 5.language_env_standalone.py      # (실습) Language Instruction이 포함된 환경
│
├── 6.visualize_data_standalone.py    # (실습) 추가 데이터/결과 시각화
│
├── 7.pi0.ipynb                       # (실습) Pi0 정책 개념 실습
├── 7.pi0_train.ipynb                 # (실습) Pi0 모델 학습
├── 7.deploy_pi0_omy.py               # (실습) Pi0 모델 배포 실행 스크립트
│
├── 8.smolvla_train.ipynb             # (실습) SmolVLA 학습
├── 8.deploy_smolvla_omy.py           # (실습) SmolVLA 배포 실행 스크립트
│
├── asset/                            # 시뮬레이션에 사용되는 모든 자산 (처음부터 존재)
│   ├── example_scene_y.xml            # MuJoCo 월드 정의
│   ├── example_scene_y2.xml
│   ├── robotis_omy/                   # 로봇 모델 (OMY)
│   ├── tabletop/                      # 테이블, 환경 모델
│   └── objaverse/                     # 물체(mesh) 데이터
│
├── mujoco_env/                       # ⚙️ MuJoCo 환경 핵심 코드 (웬만하면 수정 ❌)
│   ├── mujoco_parser.py               # MuJoCo 래퍼
│   ├── ik.py                          # Inverse Kinematics 구현
│   ├── transforms.py                  # 좌표/회전 변환
│   ├── utils.py                       # 시각화, 유틸 함수
│   ├── y_env.py                       # 기본 환경
│   └── y_env2.py                      # 언어/확장 환경
│
├── requirements.txt                  # 파이썬 패키지 목록
├── README.md                          # 프로젝트 전체 설명
│
├── pi0_omy.yaml                      # Pi0 학습 설정 파일
├── smolvla_omy.yaml                  # SmolVLA 학습 설정 파일
├── train_model.py                    # ★ 공통 학습 엔트리 포인트
│
├── demo_data_example/                #  (처음부터 존재) 예제 데이터
│   ├── data/
│   └── meta/
│
├── demo_data_language/               #  [실습 후 생성] 언어 태스크 예제 데이터
│   ├── data/
│   └── meta/
│
├── demo_data/                        # [실습 후 생성] 학생들이 직접 수집한 데이터
│   ├── data/
│   └── meta/
│
├── ckpt/                             # [실습 후 생성] 학습된 모델 체크포인트
│   ├── act_y/
│   └── smolvla_omy/
│
├── media/                            # 결과 시각화 (gif, png) — 참고용
```

### 22.3 사전 준비 및 필수 설정 (Installation & Auth)

실습을 시작하기 위해 로컬 환경을 구축하고, 모델 학습 및 공유를 위한 외부 서비스 인증을 완료해야 합니다.

#### 🛠️ 로컬 환경 구축 및 설치

```
# 1. 가상환경 생성 및 활성화
conda create -n vla python=3.10
conda activate vla

# 2. 저장소 클론 및 패키지 설치
git clone https://github.com/JJinsup/lerobot-mujoco-tutorial
cd lerobot-mujoco-tutorial
pip install -r requirements.txt

# 3. 에셋 압축 해제 (시뮬레이션 환경 구성)
cd asset/objaverse
unzip plate_11.zip
cd ../.. # 프로젝트 루트로 이동
```

#### 🤗 Hugging Face 세팅

[https://huggingface.co/](https://huggingface.co/)

1. **허깅페이스 가입 후 인증**

smolVLA는 PaliGemma 기반이므로 **모델 사용 승인**이 필수입니다.

2. **모델 사용 동의**: [PaliGemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) 접속 후 'Agree and access repository' 클릭
3.  **토큰 발급**: Settings > Access Tokens에서 발급

    * **Local (데이터 업로드용)**: `Write` 권한 토큰
    * **Server (모델 다운로드용)**: `Read` 권한 토큰

    <figure><img src="../.gitbook/assets/image (41).png" alt=""><figcaption></figcaption></figure>
4.  **CLI 로그인 (로컬 & 서버 공통)**:

    ```
    conda activate vla
    huggingface-cli login
    # 발급받은 토큰 붙여넣기 (화면에 표시되지 않음)

    # 로그인 확인
    huggingface-cli whoami
    ```

#### 📊 Weights & Biases (W\&B) 세팅

학습 로그를 실시간으로 모니터링하기 위해 사용합니다.

1. **계정 생성**: [wandb.ai](https://wandb.ai/authorize) 가입 (학교 이메일 권장)
2. **API Key 복사**: 가입 후 표시되는 API Key를 복사해둡니다.
3.  **서버 로그인**:

    ```
    pip install --upgrade wandb  # 필요한 경우
    wandb login
    # 복사한 API Key 붙여넣기
    ```

### 22.4 ipynb vs standalone.py 사용 원칙

저장소에는 같은 기능을 하는 파일이 두 종류로 존재합니다. 상황에 맞는 올바른 파일을 선택하세요.

| 파일 종류                 | 역할 및 특징             | 추천 용도              |
| --------------------- | ------------------- | ------------------ |
| **.ipynb**            | 코드의 단계별 설명 및 시각화 포함 | 구조를 "읽고 이해"하는 용도   |
| **\*\_standalone.py** | 독립 실행 가능한 파이썬 스크립트  | **실습 표준 (실제 실행용)** |

{% hint style="info" %}
**주의 사항: Jupyter Kernel Error** MuJoCo 렌더링 특성상 Jupyter Notebook에서 커널 충돌이나 에러가 매우 자주 발생합니다. 실습 중 문제가 생기면 **ipynb를 고집하지 말고 즉시 `standalone.py`로 실행**하세요.&#x20;
{% endhint %}

### 22.5 두 가지 정책(Policy)의 관계

이번 실습에서는 **ACT**, **smolVLA** 두 가지 모델을 다루지만, 각각의 학습 목적이 다릅니다.

| 정책 (Policy) | 특징              | 이번 실습에서의 역할                        |
| ----------- | --------------- | ---------------------------------- |
| **ACT**     | 구조가 단순하고 학습이 빠름 | 전체 파이프라인 완주 및 이해용                  |
| **smolVLA** | 최신 경량 VLA       | **최종 목표**: Language → Action 구조 확인 |

* **ACT**: 처음부터 끝까지 한 번 "맛보기" 위한 모델
* **smolVLA**: 언어 명령어가 로봇의 동작으로 연결되는 과정을 실제로 구현

{% hint style="info" %}
**pi0**: "대규모 VLA는 이런 식이다"라는 것을 경험하면 좋지만 학습이 너무 오래 걸리고 메모리 부족으로 스킵
{% endhint %}

### 🏁 다음 장 예고

* **\[23] MuJoCo: Train & Deploy ACT** → ACT를 이용해 데이터 수집부터 추론까지 전체 파이프라인을 완주합니다.
* **\[24] MuJoCo: Train & Deploy smolVLA** → Language 조건이 들어간 VLA 구조로 확장하여 더 지능적인 로봇 제어를 실습합니다.
