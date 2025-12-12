---
description: >-
  기본 예제(Cart-Pole)이 아닌 실제 상용 로봇(TurtleBot3)을 시뮬레이션 환경에 불러오고, 주변에 물체(Object)를
  배치하여 나만의 실험실을 꾸미는 방법을 배웁니다.
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
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/basics/integrations
---

# 🤖 \[7] MuJoCo: MJCF로 커스텀 로봇 만들기

### 🎯 학습 목표

1. **MJCF (MuJoCo XML)** 구조를 이해하고 태그별 역할을 파악합니다.
2. **MuJoCo Viewer**를 통해 로봇 모델과 환경을 시각적으로 확인합니다.
3. **프로젝트 폴더 구조**를 파악하고, Python 스크립트로 시뮬레이션을 제어합니다.

### 1. MJCF (MuJoCo XML File) 이해하기

**MJCF**는 MuJoCo에서 시뮬레이션 환경과 로봇을 정의하는 표준 포맷입니다. HTML처럼 태그(Tag) 구조로 되어 있습니다. 여러분의 로봇도 이 포맷으로 정의되어 있어야 합니다.

#### 📜 주요 태그 (Key Tags)

<table data-header-hidden><thead><tr><th width="131"></th><th width="424"></th><th></th></tr></thead><tbody><tr><td>태그</td><td>역할</td><td>비고</td></tr><tr><td><code>&#x3C;mujoco></code></td><td>파일의 시작과 끝을 알리는 루트 태그</td><td>모델 이름 정의</td></tr><tr><td><code>&#x3C;compiler></code></td><td>각도 단위(degree/radian), 메쉬 파일 경로 설정</td><td>컴파일 옵션</td></tr><tr><td><code>&#x3C;option></code></td><td>중력, 시간 간격(timestep), 마찰 등 물리 속성 설정</td><td>전역 설정</td></tr><tr><td><code>&#x3C;worldbody></code></td><td><strong>가장 중요!</strong> 실제 로봇, 바닥, 물체 등을 정의하는 공간</td><td>물리적 실체</td></tr><tr><td><code>&#x3C;asset></code></td><td>3D 모델(STL/OBJ), 텍스처, 재질 등을 미리 불러오는 곳</td><td>자원 관리</td></tr><tr><td><code>&#x3C;actuator></code></td><td>모터나 실린더 같은 구동기(Actuator) 정의</td><td>제어 입력 연결</td></tr><tr><td><code>&#x3C;sensor></code></td><td>카메라, LiDAR, IMU 등 센서 정의</td><td>데이터 수집</td></tr></tbody></table>

### 2. 실습 로봇 준비: TurtleBot3

이번 실습에서는 로봇 연구 및 교육용으로 널리 쓰이는 **TurtleBot3 (Burger)** 모델을 사용하겠습니다. 로보티즈(ROBOTIS)에서 공식적으로 제공하는 MuJoCo용 모델(MJCF)을 다운로드 받아봅시다.

> **🔗 출처:** [ROBOTIS-GIT/robotis\_mujoco\_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie?tab=readme-ov-file)

<figure><img src="../.gitbook/assets/image (18).png" alt=""><figcaption></figcaption></figure>

#### 1) 로봇 모델 다운로드 (Git Clone)

터미널을 열고 작업 디렉토리에서 아래 명령어를 입력하여 로봇 모델을 다운로드합니다.

```
# 로보티즈 MuJoCo 저장소 클론
git clone [https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git)
```

다운로드가 완료되면 `asset/robotis_tb3` 경로에 우리가 사용할 로봇 파일들이 위치하게 됩니다.

### 3. 프로젝트 폴더 구조 및 역할 (Structure)

성공적인 로봇 시뮬레이션을 위해서는 파일들이 어디에 있는지 정확히 알아야 합니다. 우리가 작업할 폴더 구조는 크게 **Asset(재료), Scripts(요리사), Utils(조리도구)** 세 가지로 나뉩니다.

```
.
├── asset                  # 로봇 모델과 3D 파일들
│   └── robotis_tb3
│       ├── assets         # STL 메쉬 파일 (로봇 껍데기)
│       ├── turtlebot3_burger.xml  # 터틀봇 MJCF 모델 파일
│       ├── factory.xml  # 환경 파일
│       └── tb3_factory.xml # 환경 + 터틀봇 통합 파일
│
├── scripts                # 메인 실행 코드
│   └── tb3_tutorial.py    # 우리가 실행할 메인 파일
│
└── utils                  # 유틸 함수들
    ├── scene_creator.py   # XML 파일들을 합쳐주는 역할
    ├── mujoco_renderer.py # 화면 렌더링을 담당
    └── ...
```

#### 📂 1) Asset 폴더 (`asset/`)

시뮬레이션의 "재료"들이 모여있는 곳입니다.

* **`turtlebot3_burger.xml`**: 로봇의 뼈대, 관절, 모터 정보가 정의된 파일입니다.
* **`assets/*.stl`**: 로봇의 생김새를 담당하는 3D 메쉬 파일들입니다.
* **`scene_*.xml`**: 로봇뿐만 아니라 바닥, 조명까지 합쳐진 완성된 무대 파일입니다.

#### 📂 2) Utils 폴더 (`utils/`)

복잡한 기능을 미리 만들어둔 "도구"들입니다. 우리가 바퀴를 다시 발명할 필요가 없게 해주죠.

* **`scene_creator.py`**: 여러 개의 XML 파일(로봇, 바닥, 물체)을 레고처럼 조립해서 하나의 월드(Scene)로 만들어주는 핵심 도구입니다.
* **`mujoco_renderer.py`**: 시뮬레이션 화면을 띄우고, 마우스/키보드 입력을 처리하는 복잡한 코드가 들어있습니다.

#### 📂 3) Scripts 폴더 (`scripts/`)

우리가 실제로 작업할 "작업대"입니다.

* **`tb3_tutorial.py`**: `asset`에서 로봇을 가져오고, `utils`의 도구를 사용해서 시뮬레이션을 돌리는 **메인 코드**입니다. 우리는 이 파일을 수정하고 실행합니다.

### 4. 파이썬으로 동적 환경 구성하기

이제 `tb3_factory.xml`로 만든 터틀봇3와 공장 환경을 파이썬 코드로 직접 불러와서 시뮬레이션을 돌리는 단계입니다.

여기서는 복잡한 조립 과정 없이, **완성된 MJCF(scene)를 그대로 로드해서 실행하는 구조**를 사용합니다.

#### 4.1 흐름 이해하기

이번 파트의 핵심 아이디어는 딱 세 가지입니다.

1. **MJCF(scene) 분리 설계:**
   * `factory.xml`: 공장, 바닥, 벽, 조명 등 "배경"
   * `tb3_burger_sensor.xml`: 센서가 장착된 터틀봇3 버거
   * `tb3_factory.xml`: 위 둘을 `<include>`로 합친 **"최종 Scene"**
2. **파이썬에서는 최종 Scene만 로드:**
   * 더 이상 파이썬 코드 내에서 XML을 합치지 않습니다.
   * `mj.MjModel.from_xml_path("tb3_factory.xml")` 한 줄로 전체 환경을 불러옵니다.
3. **MuJoCoViewer 재사용:**
   * `utils/mujoco_renderer.py` 안의 `MuJoCoViewer`를 그대로 재사용합니다.
   * 이 뷰어는 **메인 뷰(Observer View)**, **로봇 카메라 뷰(Camera View)**, **키보드 제어**, **마우스 시점 조작**을 한 번에 처리해 줍니다.

### 5. 동작 확인하기

이제 코드를 실행하여 터틀봇을 직접 조종해보고, 센서가 제대로 작동하는지 확인해 봅시다.

#### 🎮 조작 방법 (Controls)

터미널에서 스크립트를 실행한 후, **Main View** 창을 클릭하여 포커스를 맞추고 아래 키들을 눌러보세요.

| 키 (Key) | 동작 (Action)            | 설명                       |
| ------- | ---------------------- | ------------------------ |
| **`W`** | **전진 (Forward)**       | 로봇이 앞으로 이동합니다.           |
| **`S`** | **후진 (Backward)**      | 로봇이 뒤로 이동합니다.            |
| **`A`** | **좌회전 (Turn Left)**    | 로봇이 왼쪽으로 제자리 회전합니다.      |
| **`D`** | **우회전 (Turn Right)**   | 로봇이 오른쪽으로 제자리 회전합니다.     |
| **`L`** | **LiDAR 시각화 (On/Off)** | **파란색 라이다 레이저**를 켜고 끕니다. |
| `Space` | 일시정지 (Pause)           | 시뮬레이션을 멈추거나 다시 시작합니다.    |

#### 📸 실행 결과 (Result)

코드가 정상적으로 실행되면 아래와 같이 **두 개의 창**이 동시에 떠야 합니다.

**1) Observer View (메인 뷰) & LiDAR 시각화**

전체 공장 환경을 보여주는 메인 뷰입니다. **`L` 키를 누르면** 로봇 주변으로 퍼져나가는 파란색 LiDAR 센서 레이저를 볼 수 있습니다. 오른쪽 상단에는 로봇의 IMU 데이터(자세, 각속도, 가속도)가 실시간으로 표시됩니다.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-10 15-53-21.png" alt=""><figcaption></figcaption></figure>

**2) Camera View (로봇 시점)**

터틀봇에 장착된 카메라가 보는 세상입니다. 로봇을 회전시키면 이 화면도 같이 회전하며, 바닥의 글씨나 주변 장애물을 확인할 수 있습니다. LiDAR 레이저도 카메라에 잡혀 보입니다.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-10 15-53-29 (1).png" alt=""><figcaption></figcaption></figure>

> **🎉 축하합니다!** 이제 터틀봇이 공장 안을 자유롭게 돌아다니며 세상을 인식하는 환경이 완성되었습니다. 다음 챕터부터는 LLM 실습을 진행하겠습니다.
