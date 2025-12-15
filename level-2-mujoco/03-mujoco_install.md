---
description: 물리 엔진 MuJoCo를 설치하고, 딥마인드에서 제공한 Menagerie에서 실제 로봇 모델들을 불러와 화면에 띄워보겠습니다.
layout:
  width: default
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: true
metaLinks:
  alternates:
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/basics/editor
---

# 🤖 \[3] MuJoCo: 설치 및 로봇 불러오기

### 1. MuJoCo 라이브러리 설치

터미널을 열고 우리가 만든 가상환경에 진입한 뒤, 필요한 패키지들을 설치합니다.

#### 1) 가상환경 활성화

```bash
conda activate mujoco
```

#### 2) 필수 패키지 설치

`pip`를 최신 버전으로 업데이트하고, `mujoco`와 시각화를 위한 `mediapy`, `matplotlib` 등을 함께 설치합니다.

```bash
# pip 업그레이드 (에러 방지용)
pip install --upgrade pip

# MuJoCo 및 실습용 라이브러리 설치
pip install mujoco-python-viewer glfw mediapy ipywidgets control opencv-python matplotlib

# ffmpeg install
sudo apt install ffmpeg -y
```

#### 3) 설치 확인 (기본 뷰어 실행)

설치가 잘 되었는지 확인하기 위해 기본 뷰어를 실행해 봅니다.

```bash
python -m mujoco.viewer
```

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-05 16-49-10.png" alt=""><figcaption></figcaption></figure>

### 2. MuJoCo Menagerie 로봇 불러오기

구글 딥마인드에서는 유명한 로봇들의 모델 파일(XML/MJCF)을 모아둔 **\[MuJoCo Menagerie]** 라는 오픈소스 프로젝트를 운영합니다.

<figure><img src="../.gitbook/assets/image (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/image (3) (1) (1).png" alt=""><figcaption></figcaption></figure>

#### 1) Git 설치 및 저장소 클론

먼저 로봇 데이터들을 내 컴퓨터로 다운로드 받습니다.

```bash
# git 설치 (이미 있다면 생략 가능)
sudo apt install git -y

# Menagerie 저장소 다운로드
git clone [https://github.com/google-deepmind/mujoco_menagerie.git](https://github.com/google-deepmind/mujoco_menagerie.git)
```

#### 2) 유명한 로봇들 소환해보기! 🤖

다운로드 받은 폴더(`mujoco_menagerie`) 안에 있는 로봇들을 뷰어로 실행해 봅시다.

{% hint style="info" %}
**Menagerie 폴더에서 실행**
{% endhint %}

**🦾 SO-ARM 100 (실습할 로봇팔과 유사)**

```bash
python -m mujoco.viewer --mjcf trs_so_arm100/scene.xml
```

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-05 16-53-06.png" alt=""><figcaption></figcaption></figure>

**🐕 Unitree Go2 (4족 보행 로봇)**

```bash
python -m mujoco.viewer --mjcf unitree_go2/scene.xml
```

**🤖 Unitree G1 (휴머노이드)**

```bash
python -m mujoco.viewer --mjcf unitree_g1/scene.xml
```

**👨‍🍳 Aloha (양팔 로봇)**

```bash
python -m mujoco.viewer --mjcf aloha/scene.xml
```

### 🎮 뷰어 조작 및 관절 제어 실습

#### 🕹️ 기본 조작 (Spacebar & Mouse)

* **`Spacebar` :** 시뮬레이션을 **일시 정지(Pause)** 하거나 다시 **재생(Run)** 합니다.
  * _로봇이 축 늘어지거나 멈춰있다면 스페이스바를 눌러보세요._
* **마우스 우클릭 드래그:** 카메라 시점 회전
* **마우스 휠:** 화면 확대 / 축소

#### 🎛️ 우측 패널: 관절 정밀 제어 (Control Panel)

화면 **오른쪽**에 있는 **Control** 메뉴가 바로 로봇 조종기입니다.\
여기에 있는 슬라이더들을 마우스로 움직여보세요.

* **Rotation / Pitch / Elbow:**
  * 로봇 팔의 각 관절(Joint)에 해당하는 슬라이더입니다.
  * 바를 좌우로 드래그하면 해당 관절이 모터 힘에 의해 `윙-` 하고 돌아갑니다.
* **Jaw (또는 Gripper):**
  * 로봇 손(그리퍼)을 벌리거나 오므리는 슬라이더입니다.
* **수치 입력:**
  * 슬라이더 옆의 숫자 박스를 클릭하면 원하는 각도를 직접 입력할 수도 있습니다.

> **💡 팁:** 만약 슬라이더를 움직여도 로봇이 반응하지 않는다면, `Spacebar`를 눌러서 시뮬레이션이 '재생 중(Run)'인지 확인하세요! 정지 상태에서는 명령이 먹히지 않습니다.

#### 🦾 마우스로 잡아당기기 (Perturbation)

* **관절 강제 제어 (Ctrl + 우클릭 드래그):**
  * 키보드 **`Ctrl`** 키를 누른 상태에서 로봇 팔을 **우클릭한 채로 잡아당겨(Drag)** 보세요.
  * 투명한 고무줄이 생긴 것처럼 로봇을 강제로 끌어당길 수 있습니다. 로봇이 버티려고 힘을 쓰는 모습을 관찰해 보세요.

> 이제 멋진 로봇들을 내 컴퓨터에 불러올 수 있게 되었습니다! 다음 페이지에서는 Python 코드로 직접 물리 법칙을 만들고 카메라를 움직여보겠습니다.
