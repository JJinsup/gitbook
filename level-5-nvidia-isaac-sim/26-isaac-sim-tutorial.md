---
description: >-
  Isaac Sim은 사용자의 목적과 개발 단계에 따라 시뮬레이션을 제어할 수 있는 세 가지 주요 워크플로우를 제공합니다. 각 방식의 특징을
  이해하고 상황에 맞는 도구를 선택하는 것이 중요합니다.
icon: user-robot
---

# \[26] Isaac sim: Tutorial

### 1. Isaac Sim 조작의 3가지 방법 (Workflows)

#### 1.1 GUI (Graphical User Interface)

* **설명**: Omniverse Kit 기반의 그래픽 인터페이스를 직접 사용하여 시각적으로 조작하는 가장 기본적인 방식입니다.
* **특징**:
  * 상단 메뉴바(Create, File 등)와 도구 모음(Gizmo)을 이용해 물체를 배치, 이동, 회전, 스케일링할 수 있습니다.
  * 마우스 클릭만으로 물리 속성(Rigid Body, Collision)을 부여하고 시뮬레이션을 즉시 실행할 수 있습니다.
* **용도**: 초보자의 입문 단계나 복잡한 씬(Scene)의 초기 구성 단계에 적합합니다.

#### 1.2 Extensions (확장 기능/스크립트 에디터)

* **설명**: 시뮬레이터 내부의 **Script Editor** 등을 사용해 Python 코드로 환경을 실시간 제어하는 방식입니다.
* **특징**:
  * GUI가 켜진 상태에서 실시간으로 코드를 실행하여 씬을 수정할 수 있습니다.
  * **Hot-reloading** 기능을 지원하여, 코드를 수정하고 저장하면 앱을 재시작할 필요 없이 즉시 반영됩니다.
  * `Window > Examples > Robotics Examples`에서 예제 코드를 열어 즉시 확인할 수 있습니다.
* **용도**: 시뮬레이터 내부에 커스텀 UI 버튼을 만들거나, 복잡한 로직을 가진 내부 앱을 개발할 때 사용됩니다.

#### 1.3 Standalone Python (독립형 파이썬 스크립트)

* **설명**: Isaac Sim 앱 외부(터미널)에서 `./python.sh`를 통해 스크립트를 실행하여 시뮬레이션을 구동하는 방식입니다.
* **특징**:
  * 시뮬레이터 자체를 하나의 파이썬 라이브러리처럼 호출하여 사용합니다.
  * 그래픽 화면이 필요 없는 **Headless 모드** 실행이 가능하며, 터미널에서 `python my_script.py` 형태로 실행합니다.
  * 예제 위치: `<isaac-sim-root-dir>/standalone_examples/tutorials/`
* **용도**: 강화 학습(Isaac Lab), 대규모 데이터 생성(SDG), 자동화된 테스트 및 배포 단계에서 필수적입니다.

### 2. Quick Tutorials v5.1.0 실습 (GUI 기반)

#### 실행

```
# Run Script
cd /data2/[본인계정]/isaac/isaacsim

# ssh version
./isaac-sim.streaming.sh
```

텅 빈 스테이지에서 로봇을 움직이는 단계까지의 가장 기초적인 실습 과정입니다.

#### 2.1 Isaac Sim 기초 사용법 (Basic Usage)

1.  **환경 구성 (Add Ground, Light, Cube)**

    * **New Stage**: `File > New`
    * **바닥**: `Create > Physics > Ground Plane`
    * **조명**: `Create > Lights > Distant Light`
    * **물체**: `Create > Shape > Cube`

    <figure><img src="../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>
2. **조작법 익히기**
   * **W 키**: 이동 (Move) - 화살표 드래그
   * **E 키**: 회전 (Rotate) - 원 드래그
   * **R 키**: 크기 (Scale) - 네모 드래그
3. **물리 법칙 적용 (Physics & Collision)**
   * **대상 선택**: Stage 패널에서 `Cube` 선택
   * **속성 추가**: Property 패널 하단의 **+ Add** 버튼 클릭
   * **프리셋 적용**: `Physics > Rigid Body with Colliders Preset` 선택
   * **실행**: 왼쪽 툴바의 **Play (▶)** 클릭 → 큐브가 바닥으로 떨어지면 성공

#### 2.2 기초 로봇 튜토리얼 (Basic Robot)

1. **로봇 생성**
   * `File > New`로 초기화 후 `Create > Robots > Franka Emika Panda Arm` 선택
2.  로봇 살펴보기 : Examine the robot

    * `Tools > Physics > Physics Inspector` 이동 후 Franka 선택
    * 관절의 상한/하한(Limits) 및 기본 위치 확인
    * 우측 상단 아이콘(≡) 클릭 시 더 상세한 옵션 확인 가능 (파란색 Position 바를 움직여 실시간 확인)
    * 초록색 체크 표시(✔)를 클릭하면 변경사항 저장 가능

    <figure><img src="../.gitbook/assets/image (49).png" alt=""><figcaption></figcaption></figure>
3. 로봇 제어하기 : Control the Robot

* `Tools > Robotics > Omnigraph Controllers > Joint Position` 선택
* Franka를 대상으로 선택하고 `OK` 클릭하여 그래프 생성
* GUI 기반의 로봇 컨트롤러는 Omniverse의 비주얼 프로그래밍 도구인 OmniGraph 내부에 존재

{% embed url="https://docs.isaacsim.omniverse.nvidia.com/5.0.0/omnigraph/index.html#isaac-sim-omnigraph-overview-page" %}

4. **로봇 움직이기**

* Stage 탭에서 `Graph > Position_Controller` 선택 후 `JointCommandArray` 노드 선택
* Property 탭의 **Inputs** 항목(관절 명령 값)을 클릭+드래그하거나 숫자를 입력
* `Construct Array Node` 아래에 있는 **Inputs** 항목들은 로봇의 각 관절과 대응하며, 맨 아래쪽 베이스 관절부터 시작된다.
* Inputs 필드를 클릭+홀드+드래그하거나 숫자를 직접 입력하고Play (▶)하여 로봇 팔의 움직임 확인

5. **그래프 시각화**

* `Window > Graph Editors > Action Graph` 클릭
* **Edit Action Graph** 버튼으로 `Position_Controller` 선택
* 배열(Array) 노드를 선택한 뒤 **Stage** 및 **Property** 탭을 검토하여 각 배열 노드와 연결된 값들을 확인
* 그래프 안의 **Articulation Controller** 객체(노드)를 선택하여 그 속성을 검토 해보기

### 3. 핵심 튜토리얼 시리즈 요약

* **공식 튜토리얼 리스트**: [Tutorial Reference Table](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/introduction/tutorial_list.html)

| 시리즈                      | 주요 학습 내용                                 |
| ------------------------ | ---------------------------------------- |
| **Quick Start (기초)**     | 환경 구축, 물리 적용 및 로봇 조작 입문                  |
| **Core API Tutorial**    | Python API를 사용하여 프로그래밍 방식으로 환경 구축 및 제어   |
| **Robot Setup**          | 외부 CAD/URDF 파일 임포트, Joint(관절) 설정 및 로봇 조립 |
| **ROS 2 Integration**    | ROS 2 브릿지를 활용한 센서 데이터 송신 및 제어 명령 수신      |
| **Synthetic Data (SDG)** | Replicator를 사용하여 AI 학습용 합성 데이터 자동 생성     |
| **Isaac Lab**            | 로봇의 강화 학습(RL)을 위한 전문 프레임워크 활용            |
| **Sensors & Motion**     | LiDAR/카메라 시뮬레이션 및 RMPflow 등 모션 알고리즘 적용   |

### 4. 학습자를 위한 시작 가이드 팁

Isaac Sim의 구조를 가장 빠르게 이해하는 방법은 "비교 학습"입니다.

1. **GUI로 먼저 시도**: 위 실습 내용을 따라하며 마우스로 직관적인 감을 익힙니다.
2. **코드로 재현**: 동일한 작업을 `Script Editor`에서 파이썬 코드로 작성해 봅니다.
3. **Standalone으로 확장**: 최종적으로 `./python.sh` 스크립트로 작성하여 자동화 환경을 구축해 봅니다.

> **학습 포인트: USD의 이해** Isaac Sim의 모든 조작은 내부적으로 **USD(Universal Scene Description)** 데이터 구조를 변경하는 과정입니다. GUI의 버튼 클릭과 파이썬 코드 명령어는 결국 동일한 USD 속성을 수정하는 것임을 이해하면 전체 시스템 파악이 훨씬 빨라집니다.
