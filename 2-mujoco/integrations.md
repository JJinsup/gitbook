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

# 🤖 \[7] MuJoCO MJCF로 커스텀 로봇 만들기

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
│       └── scene_turtlebot3_burger.xml # 로봇 + 환경 통합 파일
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

###
