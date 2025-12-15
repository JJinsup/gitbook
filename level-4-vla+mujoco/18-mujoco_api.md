---
description: 이 문서는 MuJoCo 기반 TurtleBot3 프로젝트에서 사용되는 모든 공통 유틸리티와 런타임 구성 요소를 API 레벨에서 정의합니다.
---

# \[17] MuJoCo: API Reference

**문서의 목표**

1. 각 파일이 무슨 책임을 가지는지 명확히 합니다.
2. 메소드 단위로 Input, Output, Effect을 명시합니다.
3. `tb3_sim.py`가 어떤 유틸을 어떻게 조합하는지 한눈에 보이게 합니다.

### 전체 아키텍처 요약

이 프로젝트의 핵심 설계 철학은 다음과 같습니다.

* **Observation**은 항상 최신 프레임에서 생성됩니다.
* **Action**은 `문자열` → `제어 신호`로 변환됩니다.
* policy는 코드가 아니라 "구조"로 존재합니다.
* **LLM**은 policy를 "대체"하지 않고 **"명령 생성기"** 역할만 수행합니다.

#### Hierarchy

```
[ Input / Agent ]
   ├─ Keyboard
   ├─ LLM (Gemini)
   ↓
[ Command Queue ]
   ↓
[ TurtlebotFactorySim ]
   ├─ MuJoCo Physics
   ├─ Renderer & Camera
   ├─ YOLO Perception
   ├─ Control API
   ↓
[ Actuators (data.ctrl) ]

```

## 1. Rendering & Observation API

### Module: `mujoco_renderer.py`

#### Module Role

* MuJoCo 시뮬레이션 렌더링
* 로봇 카메라 프레임 캡처
* 사용자 입력 이벤트 처리

***

### Class: `MuJoCoViewer`

#### Description

MuJoCo 시뮬레이션을 시각화하고, 로봇 카메라 시점의 이미지를 캡처하는 고수준 렌더링 인터페이스입니다.

***

#### Method: `__init__(model, data)`

**Description**\
MuJoCo 렌더링에 필요한 window, camera, renderer, 입력 콜백을 초기화합니다.

**Parameters**

* `model` (`mj.MjModel`): MuJoCo 모델 객체
* `data` (`mj.MjData`): MuJoCo 데이터 객체

**Returns**

* None

**Effects**

* GLFW 창 2개를 생성합니다 (관찰자 뷰, 로봇 카메라 뷰).
* MuJoCo 카메라, 씬, 렌더링 컨텍스트를 초기화합니다.
* 키보드 및 마우스 콜백 함수를 등록합니다.

***

#### Method: `render_main(overlay_type=None)`

**Description**\
Observer view를 렌더링합니다.

**Parameters**

* `overlay_type (str | None)`

**Returns**

* None

**Effects**

* 메인 MuJoCo 씬을 관찰자 화면에 렌더링합니다.

***

#### Method: `render_robot()`

**Description**\
로봇 카메라 시점의 화면을 렌더링한다.

**Returns**

* None

**Effects**

* 로봇 카메라 뷰를 전용 윈도우에 렌더링합니다.

***

#### Method: `capture_img()`

**Description**\
로봇 카메라 Framebuffer를 읽어 OpenCV 호환 이미지로 반환합니다.

**Returns**

* `np.ndarray`: BGR 이미지 `(H, W, 3)`

**Effects**

* OpenGL 프레임버퍼를 읽어옵니다.
* OpenCV 호환성을 위해 픽셀 포맷 및 방향을 변환합니다.

***

#### Method: `poll_events()`

**Description**\
GLFW 이벤트를 처리한다.

**Returns**

* None

**Effects**

* 키보드, 마우스 및 윈도우 이벤트를 처리합니다.

***

#### Method: `should_close()`

**Description**\
메인 window 종료 여부를 확인한다.

**Returns**

* `bool`

***

#### Method: `terminate()`

**Description**\
렌더링 관련 모든 리소스를 해제합니다.

**Returns**

* None

**Effects**

* GLFW를 종료하고 렌더링 컨텍스트를 해제합니다.

***

## 2. Input API

### Module: `keyboard_callbacks.py`

#### Class: `KeyboardCallbacks`

***

#### Method: `keyboardGLFW(window, key, scancode, act, mods, model, scene, cam)`

**Description**\
키 입력을 받아 카메라 또는 씬 상태를 변경합니다.

**Returns**

* None

**Effects**

* 카메라 포즈(Pose) 및 상호작용 상태를 업데이트합니다.

***

### Module: `mouse_callbacks.py`

#### Class: `MouseCallbacks`

***

#### Method: `mouse_button(window, button, act, mods)`

**Effects**

* 내부 마우스 버튼 상태를 업데이트합니다.

***

#### Method: `mouse_move(window, xpos, ypos, model, scene, cam)`

**Effects**

* 커서 이동에 따라 카메라를 회전하거나 이동시킵니다.

***

#### Method: `scroll(window, xoffset, yoffset, model, scene, cam)`

**Effects**

* 카메라 뷰를 줌(Zoom) 합니다.

***

## 3. Sensor API

### Module: `lidar.py`

#### Class: `Lidar`

***

#### Method: `get_lidar_ranges(model, data, sensor_name)`

**Description**\
LiDAR 센서 거리 값을 읽습니다.

**Returns**

* `np.ndarray`: 거리 값 배열

**Effects**

* MuJoCo 센서 데이터를 읽어옵니다.

***

#### Method: `calculate_lidar_world_points(model, data, site_name, ranges)`

**Description**\
LiDAR 거리 데이터를 월드 좌표계 포인트로 변환합니다.

**Returns**

* `np.ndarray`: 월드 좌표계 포인트 배열

**Effects**

* 포인트 위치를 월드 좌표계 기준으로 계산합니다.

***

### Module: `lidar_visualizer.py`

#### Class: `LidarVisualizer`

***

#### Method: `update(model, data, sensor_name)`

**Description**\
LiDAR 데이터를 시각화합니다.

**Returns**

* None

**Effects**

* 실시간 matplotlib 플롯을 갱신합니다.

***

### 4. Perception API

#### Module: `object_detector.py`

**Module Role**

YOLO(ultralytics) 기반 객체 인식을 **런타임과 분리된 어댑터 계층**으로 제공합니다.

***

### Class: `ObjectDetector`

#### Description

YOLO 모델을 로드하고, 입력 이미지에 대해

* 구조화된 객체 탐지 결과(dict)
* 시각화된 결과 이미지를 제공하는 표준 객체 인식 인터페이스.

***

#### Method: `__init__(weight_path: str, conf: float = 0.5)`

**Description**\
YOLO 객체 탐지 모델을 초기화합니다.

**Parameters**

* `weight_path (str)`\
  YOLO 모델 가중치 파일 경로
* `conf (float)`\
  탐지 confidence threshold

**Returns**

* None

**Effects**

* YOLO 모델 가중치를 메모리에 로드한다
* 내부 추론용 스레드 락(lock)을 초기화한다
* 객체 인식 준비 상태를 구성한다

***

#### Method: `detect_dict(frame_bgr: np.ndarray)`

**Description**\
입력 이미지에서 객체를 탐지하고, 로봇 제어 및 의사결정에 사용하기 쉬운 **구조화된 관측 정보**를 반환합니다.

**Parameters**

* `frame_bgr (np.ndarray)`\
  BGR 포맷의 입력 이미지 `(H, W, 3)`

**Returns**

*   `dict[str, list[dict]]`

    ```
    {
      "label": [
        {
          "confidence": float,
          "bbox": [x1, y1, x2, y2],
          "center": [cx, cy]
        },
        ...
      ],
      ...
    }
    ```

**Effects**

* YOLO 추론(inference)을 실행합니다
* 탐지 결과를 클래스(label) 기준으로 그룹화합니다
* bounding box 및 중심 좌표를 계산합니다

***

#### Method: `detect_image(frame_bgr: np.ndarray)`

**Description**\
입력 이미지에 대해 YOLO 탐지를 수행하고, bounding box가 시각화된 이미지를 반환합니다.

**Parameters**

* `frame_bgr (np.ndarray):`BGR 포맷의 입력 이미지 `(H, W, 3)`

**Returns**

* `np.ndarray | None:`bounding box가 그려진 BGR 이미지\
  (입력이 없거나 탐지가 불가능한 경우 `None`)

**Effects**

* YOLO 추론(inference)을 실행합니다
* 탐지 결과를 입력 이미지 위에 시각적으로 렌더링합니다

***

#### Property: `available`

**Description**\
YOLO 객체 인식 기능이 사용 가능한지 여부를 반환합니다.

**Returns**

* `bool`

**Effects**

* None (상태 조회 전용)

***

## 5. Data Collection API

### Module: `camera_recorder.py`

#### Class: `CameraRecorder`

***

#### Method: `capture_and_write(frame)`

**Description**\
프레임을 영상 파일로 기록합니다.

**Returns**

* None

**Effects**

* 프레임을 디스크에 기록합니다.

***

#### Method: `release()`

**Effects**

* 비디오 파일을 마무리하고 닫습니다.

***

### Module: `video_to_frame_converter.py`

#### Module Role

* 비디오 파일을 이미지 프레임으로 분해합니다.

***

## 6. Scene Composition API

### Module: `scene_creator.py`

#### Class: `SceneCreator`

***

#### Method: `build_mjcf_scene(base_env_path, robot_path, objects_to_spawn, save_xml=None)`

**Description**\
여러 MJCF/XML 파일을 병합해 실행 가능한 씬을 생성합니다.

**Returns**

* `str`: 최종 MJCF XML

**Effects**

* XML 트리를 로드하고 수정합니다.
* 에셋 경로를 해결(Resolve)합니다.
* 선택적으로 씬 파일을 디스크에 저장합니다.

***

## 7. Control API

### Action Space

모든 제어는 의미 기반 문자열 명령으로 표현됩니다.

#### 예시

* `FORWARD`
* `LEFT`
* `STOP`
* `SEARCH_HEART`

***

#### Method: `apply_command(cmd, base_duration=1.0)`

_(tb3\_sim.py 내부)_

**Description**\
문자열 명령을 저수준 제어 신호로 변환합니다.

**Returns**

* None

**Effects**

* `data.ctrl` 값을 수정하여 액추에이터를 제어합니다.
* `current_action` 상태를 업데이트합니다.
* `action_end_sim_time`을 설정하여 자동 정지 시점을 예약합니다.

***

## 8. Runtime Orchestrator API

### Module: `tb3_sim.py`

### Class: `TurtlebotFactorySim`

#### Role

MuJoCo 기반 터틀봇 시뮬레이션의 **중앙 오케스트레이터(Runtime Orchestrator)**.\
렌더링, 물리 시뮬레이션, 객체 인식(YOLO), 명령 처리, 제어 종료 조건을 하나의 실행 루프로 결합합니다.

이 클래스는 개별 유틸리티를 구현하지 않고 각 유틸리티의 **실행 순서와 상호작용을 정의합니**다.

***

### Method: `render()`

#### Description

현재 시뮬레이션 상태를 시각적으로 렌더링하고, 로봇 카메라 기준의 최신 관측 이미지를 갱신합니다.

이 메소드는 **Observation을 생성하는 유일한 진입점**입니다.

#### Returns

* None

#### Effects

* `self.latest_frame`를 로봇 카메라의 최신 BGR 이미지로 업데이트합니다
* 관찰자(Observer) 뷰를 렌더링합니다
* 로봇 카메라 뷰를 렌더링합니다
* 키보드 및 마우스 입력 이벤트를 처리합니다

***

### Method: `step_simulation()`

#### Description

MuJoCo 물리 시뮬레이션을 설정된 FPS 기준으로 한 스텝 진행합니다.

이 메소드는 **물리 시간의 단일 전진 단위**를 정의하며, 렌더링이나 제어와 분리된 순수 물리 업데이트 역할을 수행합니다.

#### Returns

* None

#### Effects

* 시뮬레이션 시간을 흐르게 만듭니다
* MuJoCo 내부 물리 상태(`data`)를 업데이트합니다

***

### Method: `_process_commands()`

#### Description

외부 에이전트(LLM, 키보드 입력 등)가 삽입한 명령을 `command_queue`에서 순차적으로 소비하여 실행합니다.

이 메소드는 **명령 생성과 명령 실행을 분리**하기 위한 핵심 메커니즘입니다.

#### Returns

* None

#### Effects

* `command_queue`에서 대기 중인 명령을 제거합니다
* 각 명령에 대해 `apply_command()`를 호출합니다
* 현재 실행 중인 액션 상태(`current_action`)를 갱신합니다

***

### Method: `yolo_detect_dict()`

#### Description

가장 최근에 갱신된 카메라 프레임(`latest_frame`)에 대해 객체 인식 추론을 수행하고 구조화된 탐지 결과를 반환합니다.

이 메소드는 **Perception → Decision 연결 지점**으로 사용됩니다.

#### Returns

* `dict:`객체 클래스(label)를 키로 하는 탐지 결과 딕셔너리

#### Effects

* 최신 프레임에 대해 객체 인식 추론을 수행합니다
* 내부 ObjectDetector를 통해 비전 추론을 실행합니다

***

### Method: `start()`

#### Description

시뮬레이션의 메인 실행 루프를 시작합니다.

이 루프는 다음 순서를 반복합니다:

1. 명령 처리
2. 물리 시뮬레이션 스텝
3. 렌더링 및 관측 갱신
4. 검색(Search) 상태 감시
5. 액션 종료 조건 검사
6. (옵션) 객체 인식 결과 시각화

#### Returns

* None

#### Effects

* 물리 시뮬레이션, 렌더링, 제어를 지속적으로 갱신합니다
* 창 종료 또는 예외 발생 시 루프를 종료합니다
* 종료 시 `close()`를 호출하여 리소스를 정리합니다

***

### Method: `close()`

#### Description

시뮬레이션을 종료하고 모든 관련 리소스를 정리합니다.

정상 종료뿐 아니라 예외 발생 시에도 호출되어 렌더링 컨텍스트와 창이 안전하게 해제되도록 합니다.

#### Returns

* None

#### Effects

* 렌더링 및 YOLO 시각화 창을 닫습니다
* MuJoCoViewer의 렌더링 리소스를 해제합니다
* 시뮬레이션 실행 상태를 종료 상태로 전환합니다

***

## 9. LLM Agent API

### Module: `gemini_tb3.py`

### Class: `GeminiTb3`

***

#### Method: `talk(sim)`

**Description**\
자연어 입력과 감지 결과를 기반으로 로봇 명령을 생성한다.

**Returns**

* None

**Effects**

* Pushes command strings into `command_queue`

***

## 10. System-Level Summary

| 구성 요소       | 구현 방식                    |
| ----------- | ------------------------ |
| Observation | Camera frame + YOLO      |
| Action      | String-based commands    |
| Policy      | Runtime structure        |
| Reasoning   | Optional LLM             |
| Safety      | Duration-based auto stop |
