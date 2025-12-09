---
description: >-
  본 챕터에서는 기존의 터미널 기반 뷰어 실행 방식에서 벗어나, Python 바인딩을 활용하여 시뮬레이션 환경을 직접 제어하는 방법을
  학습합니다.
icon: markdown
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
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/basics/markdown
---

# \[4] Python으로 MuJoCo 다루기

### 🎯 실습 목표

1. **커스텀 물리 환경 정의:** MJCF(XML) 포맷을 이해하고, Python 코드 내에서 동적으로 물리 환경을 구성합니다.
2. **시뮬레이션 상태 조회 (Introspection):** 시뮬레이션 내의 객체(Body, Geom, Joint) ID를 식별하고, 운동학적(Kinematic) 데이터를 실시간으로 조회합니다.
3. **합성 데이터 생성 (Synthetic Data Generation):** 카메라 객체를 코드로 제어하여, 데이터셋 구축을 위한 자동화된 영상 수집 파이프라인을 구현합니다.

> VS Code에서 `.ipynb` 파일을 만들어  `Select kernel`에서 가상환경을 선택하고 아래 코드들을 순서대로 실행해 보세요.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-05 17-01-41.png" alt=""><figcaption></figcaption></figure>

### 1. 라이브러리 임포트 및 환경 변수 설정

MuJoCo를 Python 환경, 특히 Headless 서버나 Colab 환경에서 렌더링하기 위해서는 적절한 백엔드 설정이 필요합니다.

```python
import os
# MuJoCo를 GPU 없이 EGL 렌더링으로 사용하기 위한 필수 설정
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
```

### 2. MJCF(XML) 기반 물리 환경 정의

MuJoCo는 **XML(MJCF)** 형식을 통해 물리적 속성을 정의합니다. 별도의 파일을 생성하는 대신, Python 문자열을 파싱하여 모델을 로드하는 방식을 실습합니다.

정의할 환경의 구성 요소는 다음과 같습니다:

* **Worldbody:** 시뮬레이션의 루트 좌표계
* **Light:** 조명 설정 (Directional Light)
* **Geom (Floor):** 바닥 평면 (Plane)
* **Geom (Object):** 실습 대상인 캡슐(Capsule) 객체

```xml
xml = """
<mujoco>
  <worldbody>
    <!-- 위쪽에서 아래로 비추는 조명 -->
    <light pos="0 0 3" dir="0 0 -1" />
    
    <!-- 바닥 plane -->
    <geom name="floor" type="plane" size="1 1 .1" rgba=".8 .9 .8 1"/>

    <!-- 보라색 캡슐 -->
    <geom name="purple_capsule" type="capsule"
          pos=".2 .2 .8" size=".1 .2"
          rgba="0.5 0 0.5 1"/>
  </worldbody>
</mujoco>
"""
```

### 3. 모델 로드 및 초기 렌더링

작성된 XML 문자열을 MuJoCo 엔진의 **Model**과 **Data** 객체로 변환하고, 렌더링 파이프라인을 실행합니다. 이 단계에서 두 객체의 역할 차이를 이해하는 것이 중요합니다.

#### 💡 핵심 개념: MjModel vs MjData

MuJoCo는 시뮬레이션 효율성을 위해 **정적 속성**과 **동적 상태**를 철저히 분리하여 관리합니다.

**🏛️ 1. Model (`MjModel`)**

> **"변하지 않는 물리 법칙과 로봇의 뼈대"**

* **역할:** 시뮬레이션의 **정적(Static) 속성**을 정의합니다. 시간이 지나도 절대 변하지 않는 정보들입니다.
* **포함 정보:**
  * **형상 정보:** 객체의 모양(Mesh), 크기(Size), 시각적 색상(Color/Texture)
  * **물리 속성:** 질량(Mass), 관성 모멘트(Inertia), 마찰 계수(Friction)
  * **구조 정보:** 관절(Joint)의 종류 및 계층 구조(Kinematic Tree)

**⚡ 2. Data (`MjData`)**

> **"매 순간 변화하는 로봇의 현재 상태"**

* **역할:** 시뮬레이션의 **동적(Dynamic) 상태**를 저장하고 물리 엔진이 계산한 결과를 담습니다.
* **포함 정보:**
  * **운동 상태:** 현재 위치(`qpos`), 속도(`qvel`), 가속도(`qacc`)
  * **감지 정보:** 센서 측정값(Sensor data), 충돌 지점(Contacts)
  * **역학 정보:** 작용하는 힘과 토크(`xfrc_applied`), 관절에 걸리는 부하

```python
# XML 문자열 파싱 및 MjModel, MjData 인스턴스 생성
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Forward Dynamics 연산 수행 (현재 상태 갱신)
mujoco.mj_forward(model, data)

# 렌더러에 현재 Scene 상태 반영 및 픽셀 데이터 추출
renderer.update_scene(data)
frame = renderer.render()

# 결과 시각화
plt.imshow(frame)
plt.title("Initial Render Result")
plt.axis("off")
plt.show()
```

<figure><img src="../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

### 4. 모델 구조 분석 (Introspection)

로봇 제어를 위해서는 각 링크(Body)와 조인트(Joint), 형상(Geom)의 ID를 파악하여 상태 공간(State Space)에 접근해야 합니다. MuJoCo Python 바인딩을 통해 내부 객체 정보를 조회하는 방법입니다.

```python
print("--- 모델 요약 (Counts) ---")
print(f"Geometries (geom): {model.ngeom}")        # 기하학적 객체 수
print(f"Bodies (body): {model.nbody}")            # 바디(몸체) 수
print(f"Joints (joint): {model.njnt}")            # 관절 수
print(f"Actuators (actuator): {model.nu}")        # 액추에이터(구동기) 수
print(f"Sensors (sensor): {model.nsensor}")       # 센서 수

# Body 목록 출력
if model.nbody > 0:
    print("Body Names:")
    for i in range(model.nbody):
        print(f"  - {model.body(i).name}") 

# Geom 목록 출력
if model.ngeom > 0:
    print("Geom Names:")
    for i in range(model.ngeom):
        print(f"  - {model.geom(i).name}")

# 특정 geom 조회
model.geom('purple_capsule')
print('id of "purple_capsule": ', model.geom('purple_capsule').id)
print('name of geom 1: ', model.geom(1).name)
print('name of body 0: ', model.body(0).name)

# 전체 geom 목록
print("\nList of all geom names:", [model.geom(i).name for i in range(model.ngeom)])
```

### 5. 카메라 제어 및 영상 데이터 생성

고정된 시점이 아닌, 프로그램 제어를 통해 Camera Trajectory를 생성하고 영상 데이터를 수집하는 실습입니다. 이는 VLA(Vision-Language-Action) 모델 학습 시 필요한 데이터 증강(Data Augmentation)이나 **다중 시점(Multi-view) 데이터 수집**의 기초가 됩니다.

`mjvCamera` 클래스를 인스턴스화하여 카메라의 내부 파라미터(Intrinsic) 및 외부 파라미터(Extrinsic)를 프레임 단위로 갱신합니다.

```python
duration = 5         # 시뮬레이션 지속 시간 (초)
framerate = 60       # 초당 프레임 수 (FPS)
num_frames = duration * framerate
frames = []

# 커스텀 카메라 인스턴스 생성 (mjvCamera)
cam = mujoco.MjvCamera()

# 카메라 초기 파라미터 설정
cam.lookat = [0.1, 0.1, 0.1]   # 카메라가 바라볼 좌표
cam.distance = 2.0             # 카메라 거리
cam.elevation = -20            # 카메라 고도(위/아래)

# 프레임 생성
for i in range(num_frames):

    # 카메라를 부드럽게 왕복시키는 효과
    cam.distance = 2.0 + 3.0 * np.sin(i / 10)

    # 장면 업데이트 (mj_step 대신 mj_forward 사용: 정적 모델이라 step 필요 없음)
    mujoco.mj_forward(model, data)

    # Renderer에 camera 전달
    renderer.update_scene(data, camera=cam)

    # RGB 프레임 추출
    pixels = renderer.render()
    frames.append(pixels)

print("Total frames generated:", len(frames))
```

### 6. 결과 확인 및 재생

수집된 프레임 데이터를 영상으로 변환하여 시각적으로 검증합니다.

```python
media.show_video(frames, fps=framerate)
```

> 본 실습을 통해 Python API를 사용하여 **MuJoCo 물리 환경을 정의**하고, **객체 정보를 조회**하며, **카메라를 제어하여 시각적 데이터를 생성**하는 방법을 익혔습니다. 다음 챕터에서는 이를 확장하여 **진자(Pendulum) 운동**을 구현하고 시뮬레이션 해보겠습니다.
