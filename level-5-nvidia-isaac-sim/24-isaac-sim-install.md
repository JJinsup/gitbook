---
description: >-
  NVIDIA Isaac Sim은 Omniverse 위에서 동작하는 로봇 시뮬레이션 플랫폼으로, AI 기반 로봇을 “설계 – 시뮬레이션 –
  학습 – 배포”까지 한 번에 다룰 수 있도록 만든 레퍼런스 애플리케이션입니다.
icon: user-robot
---

# \[25] Isaac sim: Install

{% embed url="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html" %}

| 구분         | 내용                                                            |
| ---------- | ------------------------------------------------------------- |
| **목적**     | AI 로봇을 가상 환경에서 개발·테스트·검증                                      |
| **기반 플랫폼** | NVIDIA Omniverse, Omniverse Kit                               |
| **주요 기술**  | GPU 기반 PhysX 물리 엔진, RTX 렌더링, USD(Universal Scene Description) |
| **주요 연동**  | ROS 2, Isaac ROS, Isaac Lab, Replicator, Digital Twin 워크플로우   |

### 1. Isaac Sim의 역할

Isaac Sim은 **물리 기반 가상 환경에서 AI 로봇을 개발·시뮬레이션·테스트할 수 있게 해주는 Omniverse 기반 애플리케이션**입니다. 실제 로봇을 구동하기 전, 가상 환경에서 다음과 같은 전 과정을 처리할 수 있습니다.

1. **로봇 모델 불러오기**: URDF, CAD 등 기존 자산 가져오기
2. **센서 장착**: 카메라, LiDAR, IMU 등 가상 센서 구성
3. **물리 시뮬레이션**: 실제와 유사한 물리 법칙 적용 및 테스트
4. **학습**: RL(강화학습) 및 제어 알고리즘 최적화
5. **배포**: ROS 2/Isaac ROS를 통해 실제 로봇 하드웨어로 이식

### 2. 주요 단계별 특징

#### 2.1 Design: 로봇 및 환경 설계

Isaac Sim은 다양한 설계 데이터를 시뮬레이션 가능한 포맷으로 변환하는 강력한 파이프라인을 제공합니다.

* **지원 포맷**: URDF, MuJoCo MJCF, Onshape, CAD 기반 모델 등
* **공통 포맷 (USD)**: Pixar의 **Universal Scene Description**을 사용하여 서로 다른 툴의 자산을 동일한 기준으로 편집 및 조합 가능

#### 2.2 Tune & Train: 시뮬레이션 및 학습

고정밀 물리 엔진과 센서 시뮬레이션을 통해 **Digital Twin** 구성을 지원합니다.

* **물리/센서 시뮬레이션**: GPU 기반 PhysX 엔진으로 관절 및 접촉 시뮬레이션, RTX 기반 렌더링으로 고품질 센서 데이터(LiDAR, Depth 등) 생성
* **Replicator (합성 데이터 생성)**: 학습용 이미지를 대량 생성하고, 랜덤라이제이션(조명, 배치 등)을 통해 데이터 다양성 확보
* **Isaac Lab (RL 학습)**: 병렬 환경 시뮬레이션을 통해 강화학습 성능 극대화

#### 2.3 Deploy: 실제 로봇으로의 연결

시뮬레이션 결과물을 실제 환경으로 심리스하게 배포합니다.

* **ROS 2 Bridge**: ROS 2 토픽/서비스/액션과 직접 통신하여 HIL(Hardware-in-the-Loop) 테스트 지원
* **Isaac ROS**: NVIDIA 하드웨어 가속 패키지(SLAM, 경로 계획 등)와 연동

### 3. 시스템 아키텍처 및 개발 방식

#### 3.1 아키텍처 개요

* **Omniverse Kit 기반**: 플러그인 구조로 기능 확장이 용이하며, Python 인터프리터를 포함해 스크립팅이 자유롭습니다.
* **API 제공**: C++ 및 Python API를 제공하여 다양한 수준의 통합 개발이 가능합니다.

#### 3.2 개발 워크플로우

1. **Standalone 앱**: GUI를 사용하여 씬 구성 및 시뮬레이션 실행
2. **Python Scripting**: VS Code나 Jupyter Notebook을 연동하여 코드로 환경 및 로봇 제어
3. **ROS 2 연동**: Sim-to-Real 전이를 위한 동기/비동기 상호작용 설정

### 4. 설치 전 요구사항 (System Requirements)

Isaac Sim을 설치하기 전 다음 환경을 확인하십시오.

* **OS**: Ubuntu 20.04/22.04 또는 Windows 10/11
* **GPU**: NVIDIA RTX GPU (RTX 2070 이상 권장)
* **Driver**: 최신 NVIDIA 드라이버 (525.x 이상 권장)

### 5. 서버 설치 가이드 (Isaac Sim 5.1.0)

아래 절차는 리눅스 서버에 Isaac Sim 5.1.0 Standalone을 설치하고, 원격 스트리밍(WebRTC) 모드로 접속할 수 있도록 구성하는 과정입니다.

#### 5.1 Isaac Sim 다운로드 및 설치

```
# 설치 경로 생성
sudo mkdir -p /data2/[본인계정]/isaac
sudo chown -R [본인계정]:[본인계정] /data2/[본인계정]/isaac

# Isaac Sim Standalone 다운로드
wget "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip"

# 압축 해제
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d /data2/[본인계정]/isaac/isaacsim

# ZIP 파일 삭제
rm ./isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

#### 5.2 설치 스크립트 실행

```
cd /data/isaac/isaacsim
./post_install.sh
```

#### 5.3 Isaac Sim 실행

*   **GUI 버전 실행** (서버에 GUI가 있는 경우)

    ```
    ./isaac-sim.selector.sh
    ```
*   **SSH 환경에서 Streaming 모드 실행** (헤드리스 서버에서 주로 사용)

    ```
    ./isaac-sim.streaming.sh
    ```

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

<p align="center"><strong>Isaac sim 스트리밍 모드 준비 완료(서버)</strong></p>

### 6. 클라이언트 설정 (Isaac Sim 서버 접속)

서버에서 livestream 실행이 가능하면, 클라이언트 PC에서 WebRTC Streaming Client를 사용해 접속할 수 있습니다.

#### 6.1 방화벽 포트 열기

Isaac Sim WebRTC는 아래 포트를 사용합니다.

| 포트        | 프로토콜 | 용도          |
| --------- | ---- | ----------- |
| **49100** | TCP  | 연결·세션 관리    |
| **47998** | UDP  | 영상 스트리밍 데이터 |

**방화벽 설정 예시:**

```
# TCP 포트 허용
sudo ufw allow 49100/tcp
# UDP 포트 허용
sudo ufw allow 47998/udp
# 설정 적용
sudo ufw reload

# 포트 정상 확인
sudo netstat -tulpn | grep 49100
```

#### 6.2 WebRTC Streaming Client 설치

1. 클라이언트(내 PC)에서 [NVIDIA 공식 다운로드 링크](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html#isaac-sim-latest-release)로 이동하여 **Isaac Sim WebRTC Streaming Client**를 다운로드합니다.
2. `chmod +x isaacsim-webrtc-streaming-client-1.1.4-linux-x64.AppImage` 실행 권한을 부여합니다
3. 설치 후 실행하여 서버 주소(예: `SERVER_IP`)를 입력해 접속합니다.

<figure><img src="../.gitbook/assets/스크린샷 2026-01-16 120937.png" alt=""><figcaption></figcaption></figure>

### 7. Isaac Sim 온라인 강의 수강 (NVIDIA Developer)

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

Isaac Sim 관련 공식 온라인 강의와 튜토리얼은 NVIDIA Developer Program에 가입하면 무료로 이용할 수 있습니다.

#### 7.1 NVIDIA Developer 가입

* **가입 링크**: [https://developer.nvidia.com/](https://developer.nvidia.com/)

#### 7.2 주요 학습 자료

* **Isaac Sim 공식 튜토리얼**: [링크](https://docs.isaacsim.omniverse.nvidia.com/)
  * Quickstart, Python API 예제, ROS 2 연결, Isaac Lab(RL) 가이드
* **NVIDIA On-Demand (영상 강의)**: [링크](https://www.nvidia.com/en-us/on-demand/)
  * 검색창에 “Isaac Sim”, “Omniverse”, “Isaac Lab” 입력 시 컨퍼런스 및 워크숍 영상 시청 가능
* **NVIDIA Robotics Learning Path:**  [링크](https://www.nvidia.com/en-us/learn/learning-path/robotics/)

> **Isaac Sim 핵심 포인트**
>
> * **통합 플랫폼**: 설계부터 배포까지 하나의 워크플로우로 연결
> * **고정밀 시뮬레이션**: PhysX와 RTX 렌더링 기반의 고성능 Digital Twin
> * **대규모 학습 지원**: Replicator와 Isaac Lab을 통한 대량의 합성 데이터 및 RL 학습
> * **강력한 생태계**: USD 및 ROS 2 기반의 표준화된 연동 체계
