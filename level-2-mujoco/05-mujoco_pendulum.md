---
description: 정지된 화면이 아닌, 시간에 따라 움직이는 물리 시뮬레이션을 구현합니다.
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
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/basics/images-and-media
---

# 🤖 \[5] MuJoCo: Pendulum 시뮬레이션

### 🎯 실습 목표

Simple Pendulum 모델을 통해 MuJoCo의 **Dynamics** 계산 원리를 이해하고, 시뮬레이션 데이터를 그래프로 분석하는 방법을 익힙니다.

1. **Hinge Joint 모델링:** 회전 관절(Hinge)을 사용하여 진자 운동을 하는 로봇 모델을 정의합니다.
2. **Physics Stepping:** `mj_step()` 함수를 사용하여 물리 엔진의 시간을 전진시키는 방법을 배웁니다.
3. **Data Analysis:** 시뮬레이션 결과(위치, 속도)를 수집하여 Matplotlib으로 시각화합니다.

> **Pre-requisites:** 이전 챕터와 동일한 환경에서 새 노트북 파일을 만들어 실습하세요.

### 1. 환경 설정 및 라이브러리 준비

기본적인 설정은 이전과 동일합니다.

```python
import os
# 렌더링을 GPU 없이 headless로 하기 위한 설정 (주피터/리눅스 서버에서 필수)
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import mediapy as media
import matplotlib.pyplot as plt
```

### 2. 진자(Pendulum) 모델링 (XML)

이번에는 단순한 캡슐이 아닙니다. 관절(Joint)이 있어 움직일 수 있는 모델을 정의합니다.

```
simple_pendulum 모델 구조:
- hinge joint 1개 (회전 자유도 1)
- sphere(고정점) + capsule(막대) 형태
- 고정 anchor 위치에서 아래로 매달린 구조
```

* **`<joint type="hinge">`**: 경첩처럼 한 축으로만 회전하는 관절입니다.
* **`damping`**: 관절의 마찰(저항)을 의미합니다. 이 값이 없으면 진자는 영원히 멈추지 않습니다.
* **`timestep="0.01"`**: 시뮬레이션의 시간 간격(dt)입니다. 0.01초 단위로 물리 계산을 수행합니다.

```xml
xml = """
<mujoco model="simple_pendulum">
    <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>

    <!-- 장면 배경 흐림 효과 -->
    <visual>
        <rgba haze="1 1 1 1"/>
    </visual>

    <worldbody>
        <!-- 위에서 아래로 비추는 조명 -->
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

        <!-- 시점을 고정하는 카메라 -->
        <camera name="track" mode="fixed" pos="0 -3.5 2.2" xyaxes="1 0 0 0 1 2"/>

        <!-- 펜듈럼 고정점 -->
        <body name="anchor" pos="0 0 1.5">
            <geom type="sphere" size="0.05" rgba=".2 .2 .2 1"/>

            <!-- 실제 pendulum body -->
            <body name="pole" pos="0 0 0">
                <!-- 회전 조인트 -->
                <joint name="swing_hinge" type="hinge" axis="0 1 0" damping="0.7"/>

                <!-- 아래로 길게 이어지는 막대 -->
                <geom name="pole_geom" type="capsule"
                        fromto="0 0 0 0 0 -1.0"
                        size="0.045" rgba="0.9 0.2 0.2 1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
```

### 3. 시뮬레이션 로직 구현

가장 중요한 부분입니다. `mj_step()` 함수를 루프(Loop) 안에서 반복 호출하여 물리 세계의 시간을 흐르게 합니다.

#### 💡 `mj_forward` vs `mj_step`

* **`mj_forward(model, data)`**: 시간을 흐르게 하지 **않습니다**. 현재 상태에서 힘과 가속도만 계산합니다. (사진 촬영용)
* **`mj_step(model, data)`**: 물리 법칙에 따라 시간을 `timestep`만큼 **전진시킵니다**. 위치와 속도가 변합니다. (동영상용)

```python
def record_frame_and_data(renderer, data, frames_list, times_list, positions_list,
                          velocities_list, camera_name="track"):
    """
    현재 MuJoCo 상태를 이미지 + 상태벡터로 기록한다.
    renderer.update_scene --> 장면 업데이트
    renderer.render()     --> RGB 이미지 획득
    """
    renderer.update_scene(data, camera=camera_name)
    pixels = renderer.render()

    frames_list.append(pixels)
    times_list.append(float(data.time))
    positions_list.append(float(data.qpos[0]))   # hinge joint angle
    velocities_list.append(float(data.qvel[0]))  # hinge joint angular velocity


def simulate_pendulum(xml, duration=10.0, framerate=60, theta0=0.5):
    """
    pendulum을 duration 동안 시뮬레이션한다.
    
    - mj_step() 사용: 물리 시뮬레이션 시간에 따라 진행
    - mj_forward()는 단순 계산(정적 업데이트) 용도
    - theta0 : 초기 각도 (rad)
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames, times, positions, velocities = [], [], [], []

    # 초기 각도 설정
    data.qpos[0] = theta0  

    # 시뮬레이션 루프 (MuJoCo 내부 timestep 기반)
    while data.time < duration:
        mujoco.mj_step(model, data)

        # 'fps * 시간' 기준으로 필요한 프레임 수 계산
        if len(frames) < data.time * framerate:
            record_frame_and_data(renderer, data, frames, times, positions, velocities)

    return frames, times, positions, velocities



```

### 4. 데이터 시각화 (Plotting)

수집된 데이터를 통해 진자의 운동을 그래프로 확인해 봅니다. 위치(각도)와 속도가 주기적으로 변하는 것을 볼 수 있습니다.

```python
def plot_results(times, positions, velocities):
    """펜듈럼 상태를 시각화 (단진자라 상태가 2개)"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Pendulum Position and Velocity over Time', fontsize=16)

    axs[0].plot(times, positions, label='Angle')
    axs[0].set_title('Pendulum Position')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Position [rad]')
    axs[0].grid(True)

    axs[1].plot(times, velocities, 'r', label='Angular velocity')
    axs[1].set_title('Velocity')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
```

### 5. 실행 및 결과 확인

이제 모든 코드를 조립하여 실행해 봅시다!

```python
duration = 10   # 10초 동안 시뮬레이션
framerate = 60  # 60 FPS로 영상 저장

frames, times, positions, velocities = simulate_pendulum(
    xml, duration=duration, framerate=framerate, theta0=0.5
)

print("Simulation finished. Plotting results...")
plot_results(times, positions, velocities)
```

<figure><img src="../.gitbook/assets/image (3) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

* **감쇠 진동(Damped Oscillation):** XML에서 `damping="0.1"`로 설정했기 때문에, 시간이 지날수록 진폭이 점점 줄어드는 것을 확인할 수 있습니다. (마찰이 없다면 영원히 같은 높이로 움직였을 것입니다.)

```python
print("Rendering video...")
media.show_video(frames, fps=framerate)
```

<figure><img src="../.gitbook/assets/image (5) (1).png" alt=""><figcaption></figcaption></figure>

> **Summary:** `mj_step`을 통해 물리 법칙이 적용된 세계를 구현했습니다. 이 원리는 복잡한 휴머노이드 로봇이나 로봇 팔을 제어할 때도 똑같이 적용됩니다. 다음 시간에는 드디어 **강화학습(RL)** 환경을 구축해 보겠습니다.
