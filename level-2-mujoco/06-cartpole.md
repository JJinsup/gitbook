---
description: >-
  로봇 제어 이론의 가장 고전적이고 중요한 예제인 Cart-Pole (Inverted Pendulum, 역진자) 시스템을 강화학습으로
  다룹니다.
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
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/basics/interactive-blocks
---

# 🤖 \[6] MuJoCo: Cart-Pole 제어와 강화학습

### 🎯 실습 목표

우리는 복잡한 수학적 제어 이론(LQR 등)을 직접 푸는 대신, **MuJoCo 시뮬레이션**과 **PPO 강화학습**을 결합하여 로봇이 스스로 균형 잡는 법을 학습하도록 만듭니다.

1. **동역학 모델링 (Modeling):** 불안정한 역진자 시스템이 MuJoCo 물리 엔진에서 어떻게 구현되는지 이해합니다.
2. **상태 관측 (State Observation):** 제어를 위해 시스템의 상태(State)를 정의하고 관측하는 방법을 익힙니다.
3. **강화학습 환경 구축 (RL Environment):** 물리 엔진을 AI 학습을 위한 표준 인터페이스(Gymnasium)로 래핑(Wrapping)합니다.
4. **보상 함수 설계 (Reward Shaping):** 제어 목표(균형 유지)를 수학적인 보상 함수로 변환하고 PPO 알고리즘으로 학습합니다.

> **Pre-requisites:** `stable-baselines3`와 `gymnasium` 라이브러리가 필요합니다. 설치가 안 되어 있다면 아래 명령어로 설치하세요. `pip install gymnasium stable-baselines3`

### 1. Cart-Pole 시스템의 물리학과 MuJoCo 모델링

#### 1.1 역진자(Inverted Pendulum)란?

일반적인 진자는 중력에 의해 아래로 축 처져 안정된 상태를 유지하려 합니다. 하지만 **역진자**는 막대(Pole)가 위를 향해 서 있는 상태로, 중력에 의해 끊임없이 쓰러지려 하는 **본질적으로 불안정한(Unstable)** 시스템입니다.

우리는 막대를 직접 잡을 수 없고, 막대가 연결된 카트(Cart)를 좌우로 밀어서 그 관성력으로 막대의 균형을 잡아야 합니다. 이를 **과소구동(Underactuated) 시스템**이라고 합니다. (제어 입력은 1개인데, 제어해야 할 자유도는 카트 위치와 막대 각도 2개임)

#### 1.2 환경 설정 및 라이브러리

```python
import os
os.environ['MUJOCO_GL'] = 'egl'   # mujoco import 전에!
 
import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
```

#### 1.2 MuJoCo XML 정의

MuJoCo에서는 이 물리 시스템을 다음과 같이 정의합니다.

* **Slide Joint:** 카트가 레일 위에서 직선운동을 하도록 구속합니다.
* **Hinge Joint:** 막대가 카트 위에서 회전운동을 하도록 구속합니다.
* **Actuator (Motor):** 슬라이드 조인트(카트)에만 힘(Force)을 가할 수 있습니다.

```python
xml = """
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>
    <default>
        <joint armature="0" damping="1" limited="true"/>
        <geom friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
        <tendon/>
        <motor ctrlrange="-3 3"/>
    </default>
    <option gravity="0 0 -9.81" integrator="Euler" timestep="0.01"/>
    <size nstack="3000"/>
    <worldbody>
        <!-- 레일 (바닥) -->
        <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1"
              size="0.02 1" type="capsule" contype="0" conaffinity="0"/>
        
        <!-- 움직이는 카트 (Cart) -->
        <body name="cart" pos="0 0 0">
            <!-- 슬라이드 조인트: 직선 운동 -->
            <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0"
                   range="-1.2 1.2" type="slide"/>
            <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0"
                  size="0.1 0.1" type="capsule"/>
            
            <!-- 카트 위의 막대 (Pole) -->
            <body name="pole" pos="0 0 0">
                <!-- 힌지 조인트: 회전 운동 -->
                <joint axis="0 1 0" name="hinge" pos="0 0 0"
                       type="hinge" limited="false"/>
                <geom fromto="0 0 0 0.001 0 0.6" name="cpole"
                      rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
            </body>
        </body>
    </worldbody>
    <!-- 액추에이터: 카트를 미는 모터 -->
    <actuator>
        <motor ctrllimited="true" ctrlrange="-3 3" gear="10"
               joint="slider" name="slide"/>
    </actuator>
</mujoco>
"""
```

### 2. 상태(State)의 정의와 관측

제어를 하기 위해서는 현재 시스템이 어떤 상황인지 정확히 알아야 합니다. 이를 상태(State)라고 합니다. 역진자 시스템의 상태 $$x$$는 다음 4가지 요소로 정의됩니다.

$$
\text{State } x = [p, \theta, \dot{p}, \dot{\theta}]^T
$$

1. $$p$$ : 카트 위치 (Cart Position) - 레일 중심에서 얼마나 벗어났는가?
2. $$\theta$$: 막대의 각도 (Pole Angle) - 수직에서 얼마나 기울어졌는가?
3. $$\dot{p}$$: 카트의 속도 (Cart Velocity) - 얼마나 빠르게 이동 중인가?
4. $$\dot{\theta}$$: 막대의 각속도 (Pole Angular Velocity) - 얼마나 빠르게 쓰러지고 있는가?

MuJoCo는 `data.qpos`(위치)와 `data.qvel`(속도)를 통해 이 값을 실시간으로 제공합니다.

```python
def create_env():
    """MuJoCo 모델/데이터/렌더러 생성."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer


def reset_state(data, q_init=None, qv_init=None):
    """qpos, qvel 초기화. q_init, qv_init은 리스트나 np.array로 전달."""
    mujoco.mj_resetData(data.model, data)
    if q_init is not None:
        data.qpos[:len(q_init)] = q_init
    if qv_init is not None:
        data.qvel[:len(qv_init)] = qv_init
    mujoco.mj_forward(data.model, data)


def get_state(data):
    """
    상태 벡터 [x_cart, theta_pole, x_dot, theta_dot] 반환.
    qpos[0] : slider joint (cart position)
    qpos[1] : hinge joint (pole angle)
    """
    x = float(data.qpos[0])
    theta = float(data.qpos[1])
    x_dot = float(data.qvel[0])
    theta_dot = float(data.qvel[1])
    return np.array([x, theta, x_dot, theta_dot])


def rollout(model, data, renderer,
            duration=10.0, framerate=60,
            ctrl_func=None,
            q_init=None, qv_init=None):
    """
    한 에피소드 시뮬레이션:
      - ctrl_func(state, t)로 제어 입력 생성 (None이면 0 입력)
      - 상태 궤적과 프레임 기록.

    반환:
      times, states, controls, frames
    """
    reset_state(data, q_init=q_init, qv_init=qv_init)

    dt = model.opt.timestep
    times = []
    states = []
    controls = []
    frames = []

    # 메인 루프
    while data.time < duration:
        t = float(data.time)
        state = get_state(data)

        # 제어 입력 계산
        if ctrl_func is None:
            u = 0.0
        else:
            u = float(ctrl_func(state, t))

        # ctrl 적용
        data.ctrl[0] = u

        # 한 스텝 진행
        mujoco.mj_step(model, data)

        # 로깅
        times.append(t)
        states.append(state)
        controls.append(u)

        # 프레임 샘플링 (대략 framerate에 맞추기)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

    return (
        np.array(times),
        np.vstack(states),   # shape: [T, 4]
        np.array(controls),  # shape: [T]
        np.array(frames)     # shape: [N_frames, H, W, 3]
    )

def plot_states(times, states, controls=None, title="Cart-Pole States"):
    """
    states: shape [T, 4] = [x, theta, x_dot, theta_dot]
    controls: shape [T] or None
    """
    x = states[:, 0]
    theta = states[:, 1]
    x_dot = states[:, 2]
    theta_dot = states[:, 3]

    if controls is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=16)

        axs[0, 0].plot(times, x, label="Cart Position [m]")
        axs[0, 0].set_ylabel("Position [m]")
        axs[0, 0].set_title("Cart Position")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(times, x_dot, "r", label="Cart Velocity [m/s]")
        axs[0, 1].set_ylabel("Velocity [m/s]")
        axs[0, 1].set_title("Cart Velocity")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(times, theta, "g", label="Pole Angle [rad]")
        axs[1, 0].set_ylabel("Angle [rad]")
        axs[1, 0].set_title("Pole Angle")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(times, theta_dot, "m", label="Pole Angular Velocity [rad/s]")
        axs[1, 1].set_ylabel("Angular Velocity [rad/s]")
        axs[1, 1].set_title("Pole Angular Velocity")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=16)

        axs[0].plot(times, x, label="Cart Position [m]")
        axs[0].plot(times, theta, label="Pole Angle [rad]")
        axs[0].set_ylabel("Pos / Angle")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(times, x_dot, label="Cart Velocity [m/s]")
        axs[1].plot(times, theta_dot, label="Pole Angular Velocity [rad/s]")
        axs[1].set_ylabel("Vel / Ang. Vel")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(times, controls, "r", label="Control Input [N]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Force [N]")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
```

#### 2.1 제어 없는 상태 시뮬레이션

먼저 제어기 없이 막대를 살짝 기울였을 때 어떤 일이 일어나는지 관찰해 봅시다. 당연히 막대는 중력에 의해 쓰러지고, 멈춰 있던 카트도 반작용으로 약간 움직이게 됩니다.

```python
model, data, renderer = create_env()

# 초기 상태: 폴을 살짝(0.1 rad) 기울인 상태에서 시작
q_init = [0.0, 0.1]  # [x, theta]
qv_init = [0.0, 0.0]   # [x_dot, theta_dot]

duration = 10.0
framerate = 60

times, states, controls, frames = rollout(
    model, data, renderer,
    duration=duration,
    framerate=framerate,
    ctrl_func=None,     # 무제어
    q_init=q_init,
    qv_init=qv_init,
)

print("Simulation finished. Plotting states...")
plot_states(times, states, controls=None, title="Free Cart-Pole Dynamics")
```

<figure><img src="../.gitbook/assets/image (7) (1).png" alt=""><figcaption></figcaption></figure>

### 3. Gymnasium 환경 구축 및 보상 설계

<figure><img src="../.gitbook/assets/image (8) (1).png" alt=""><figcaption></figcaption></figure>

우리는 LQR 같은 수치 해석적 방법 대신, 강화학습을 사용해 제어기를 만들 것입니다. 이를 위해 MuJoCo를 Gymnasium Environment로 포장해야 합니다.

#### 3.1 보상 함수(Reward Function) 설계의 핵심

강화학습 에이전트는 **보상(Reward)을 최대화**하는 방향으로 행동을 학습합니다. 따라서 "우리가 원하는 이상적인 움직임"을 수학적으로 정의해 주어야 합니다.&#x20;

우리의 목표는 다음과 같습니다:

1. **Pole Angle**  $$(\theta \approx 0)$$ **:** 막대를 똑바로 세워라. (가장 중요)
2. **Cart Position** $$(p \approx 0)$$**:** 카트를 중앙에 위치시켜라. (레일 밖으로 나가지 않게)
3. **Control Input** $$(u \approx 0)$$ 힘을 적게 써라. (에너지 효율성 및 부드러운 움직임)

이를 수식으로 표현하면 비용 함수(Cost Function)가 되며, 보상은 `1.0 - Cost` 형태로 정의하여 비용이 낮을수록 높은 점수를 받게 합니다.

$$\text{Cost} = w_1 \theta^2 + w_2 p^2 + w_3 \dot{p}^2 + w_4 \dot{\theta}^2 + w_5 u^2$$

```python
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class CartPoleMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        # 기존 유틸 재사용
        self.model, self.data, self.renderer = create_env()
        self.dt = self.model.opt.timestep

        # 관측 공간: [x, theta, x_dot, theta_dot]
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 행동 공간: 카트에 가하는 힘 (연속, -3 ~ 3)
        self.action_space = spaces.Box(
            low=np.array([-3.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.max_steps = int(10.0 / self.dt)  # 에피소드 최대 길이 (대략 10초)
        self.step_count = 0

    def _get_obs(self):
        return get_state(self.data).astype(np.float32)

    def _get_info(self):
        return {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 초기화 시 랜덤하게 약간 기울어지게 함 (-0.15 ~ 0.15 rad)
        theta0 = self.np_random.uniform(-0.15, 0.15)
        q_init = [0.0, theta0]
        qv_init = [0.0, 0.0]
        reset_state(self.data, q_init=q_init, qv_init=qv_init)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1

        # action은 shape (1,) 이라고 가정
        u = float(np.clip(action[0], -3.0, 3.0))
        self.data.ctrl[0] = u

        # 한 스텝 진행
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        x, theta, x_dot, theta_dot = obs

        # 보상 설계 (Reward Shaping)
        # 폴이 세워져 있고 카트가 가운데에 있을수록 보상 높게
        cost = (
            1.0 * theta**2 +        # 각도 제곱 (기울어질수록 페널티)
            0.1 * x**2 +            # 위치 제곱 (중앙에서 멀어질수록 페널티)
            0.01 * x_dot**2 +       # 속도 제곱 (너무 빨리 움직이면 페널티)
            0.01 * theta_dot**2 +
            0.001 * u**2            # 제어 입력 제곱 (힘을 많이 쓰면 페널티)
        )
        reward = 1.0 - cost  # 최대 1 근처, 상태가 나쁠수록 작아짐

        # 종료 조건
        terminated = bool(
            abs(theta) > np.pi / 2.0 or  # 폴이 90도 이상 기울어지면 실패
            abs(x) > 1.2                 # 레일 밖으로 나가면 종료
        )
        truncated = bool(self.step_count >= self.max_steps)

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        # PPO 학습 중에는 호출 안 해도 되고,
        # 나중에 평가할 때 rgb_array로 프레임 뽑는 데 씀
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        return img

    def close(self):
        pass
```

### 4. PPO 강화학습 및 제어 성능 검증

이제 **PPO (Proximal Policy Optimization)** 알고리즘을 사용하여 최적의 제어 정책(Policy)을 학습합니다. PPO는 현재 상태 _**s**_&#xB97C; 입력받아 최적의 힘 _**u**_&#xB97C; 출력하는 신경망을 학습합니다.

$$
u = \pi_\phi(s)
$$

#### 4.1 학습 진행 (Training)

```python
# Gymnasium env 인스턴스 생성
env = CartPoleMuJoCoEnv()

# MlpPolicy: 관측 → MLP → 연속 행동
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
)

# 학습 스텝 수는 상황 봐가면서 늘려도 됨 (예: 100_000 ~ 300_000)
model.learn(total_timesteps=150_000)

# 학습된 모델 저장
model.save("ppo_cartpole_mujoco")
```

#### 4.2 제어 결과 확인 (Evaluation)

학습된 모델이 실제로 막대를 세울 수 있는지 확인합니다.

* **성공적인 제어의 기준:**
  * **Pole Angle 그래프:** 0도 근처에서 미세하게 진동하며 유지되어야 합니다.
  * **Cart Position 그래프:** 중앙(0) 근처에 머물러야 합니다.
  * **Action 그래프:** 막대가 쓰러지려 할 때마다 적절한 힘이 가해지는지 확인합니다.

```python
eval_env = CartPoleMuJoCoEnv()
obs, info = eval_env.reset()

duration = 10.0
framerate = 60
max_steps = int(duration / eval_env.dt)

frames = []
times = []
states = []
controls = []

print("Running Evaluation Episode...")
for step in range(max_steps):
    # 학습된 정책(AI)에게 현재 상태를 보여주고 행동(Action)을 결정하게 함
    # deterministic=True: 확률적 탐색을 끄고 가장 좋은 행동만 선택
    action, _ = model.predict(obs, deterministic=True)

    # 환경에 행동 적용
    obs, reward, terminated, truncated, info = eval_env.step(action)

    # 기록
    state = get_state(eval_env.data)
    states.append(state)
    controls.append(float(action[0]))
    times.append(eval_env.data.time)

    # 영상 프레임 저장
    frame = eval_env.render()
    frames.append(frame)

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

states = np.vstack(states)
controls = np.array(controls)
times = np.array(times)

print("Evaluation finished. Plotting states...")
plot_states(times, states, controls=controls, title="PPO-Controlled Cart-Pole")
```

```python
print("Rendering PPO policy video with mediapy...")
media.show_video(frames, fps=framerate)
```

#### 📈 결과 분석

<figure><img src="../.gitbook/assets/image (6) (1) (1).png" alt=""><figcaption></figcaption></figure>

그래프를 보면 초기에는 막대가 기울어져 있어($$\theta \neq 0$$) 카트가 강하게 움직이며(u 발생) 중심을 잡으려 노력합니다. 시간이 지나면 막대는 수직 상태로 수렴하고, 카트도 중앙으로 돌아오며 제어 입력(u)이 0에 가까워지는 **안정화(Stabilization)** 상태에 도달하게 됩니다. 이것이 바로 강화학습을 통해 얻은 제어기의 성능입니다.

#### 📝 학생용 과제: Cart-Pole MuJoCo 환경 분석 보고서

실습 코드를 분석하여 다음 강화학습의 핵심 구성 요소들이 코드상에서 어떻게 정의되어 있는지 구체적으로 기술하시오.

1\. State (상태, Observation)

* 에이전트(AI)가 매 순간 관측하는 정보는 무엇인가?
* 데이터의 차원(Dimension)은 몇이며, 각각의 값은 물리적으로 무엇을 의미하는가?

2\. Action (행동)

* 에이전트가 환경에 가할 수 있는 행동은 무엇인가?
* 이 행동은 이산적(Discrete)인가, 연속적(Continuous)인가?
* 행동 값의 범위(Range)는 어떻게 되는가?

3\. Reward Function (보상 함수)

* 에이전트가 높은 점수를 받기 위한 조건은 무엇인가? (목표)
* 반대로 점수가 깎이는(Penalty) 요인 4가지는 무엇인가? 코드의 `cost` 수식을 보고 해석하시오.

4\. Termination (종료 조건)

* 에피소드가 "실패"로 간주되어 즉시 종료되는 조건 2가지는 무엇인가?

5\. Algorithm (알고리즘)

* 사용된 강화학습 알고리즘의 이름은 무엇이며, 어떤 라이브러리 함수를 사용했는가?

