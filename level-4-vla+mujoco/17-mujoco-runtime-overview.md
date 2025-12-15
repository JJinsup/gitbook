---
description: 이 문서는 MuJoCo 기반 VLA 스타일 TurtleBot3 시뮬레이션의 전체 런타임 구조를 다룹니다.
---

# \[17] MuJoCo: Runtime Overview

로봇이 환경을 어떻게 인식(Perception)하고, 상황을 어떻게 판단(Decision)하며, 최종적으로 어떻게 행동(Action)하는지에 대한 논리적 흐름을 설명합니다. 이 페이지를 통해 **전체 시스템의 유기적인 연결 구조**를 파악하면, 이후 API Reference의 세부 구현 내용이 어떤 맥락에서 작성되었는지 자연스럽게 이해할 수 있을 것입니다.

### 1. 이 시뮬레이션의 목적: VLA 구조의 모사

본 프로젝트는 최신 로봇 파운데이션 모델(VLA, Vision-Language-Action)의 구조를 학습 과정 없이 **시스템 레벨에서 모사(Imitation)한 런타임**입니다.

대규모 End-to-End 모델을 직접 학습시키는 대신, **관측 → 판단 → 행동**의 각 단계를 코드 레벨에서 명시적으로 분리하고 연결하는 방식을 채택했습니다. 이를 통해 다음과 같은 이점을 얻을 수 있습니다.

1. **VLA 구조의 이해:** 블랙박스 모델 내부에서 일어나는 과정을 명시적인 모듈로 분해하여 이해할 수 있습니다.
2. **모듈의 확장성:** YOLO, LLM, RL 정책(Policy) 등 다양한 알고리즘을 동일한 런타임 위에서 실험할 수 있습니다.

<figure><img src="../.gitbook/assets/image (27).png" alt=""><figcaption></figcaption></figure>

### 2. 전체 시스템 아키텍처

시뮬레이션의 중심에는 `TurtlebotFactorySim`이라는 **Runtime Orchestrator**가 존재합니다. 이 클래스는 다음과 같은 이질적인 컴포넌트들을 하나의 실행 루프(Loop)로 묶어주는 역할을 합니다.

* **물리 엔진:** MuJoCo Physics
* **렌더링:** OpenGL 기반 렌더러
* **인식:** YOLO 기반 객체 탐지
* **입력 처리:** 키보드 및 LLM 명령 큐
* **제어 로직:** 액션 수행 및 종료 조건 관리

> **\[그림] 전체 아키텍처 블록 다이어그램**
>
> _(Input → Runtime Orchestrator → MuJoCo Physics / Renderer → Observation / Action의 구조도)_

### 3. Runtime Orchestrator의 역할

`TurtlebotFactorySim`은 개별적인 기능을 직접 구현하지 않습니다. 대신 "언제, 어떤 모듈을 호출할지"를 결정하는 관리자 역할을 수행합니다.

이 클래스의 핵심 책임은 다음과 같습니다.

1. **관측(Observation) 생성:** 센서 데이터를 수집하여 현재 상태를 정의합니다.
2. **명령(Command) 실행:** 외부(사용자/LLM)에서 들어온 명령을 물리적 동작으로 변환합니다.
3. **액션 관리:** 수행 중인 액션을 모니터링하고, 조건 만족 시 자동으로 종료합니다.
4. **상태 관리:** 탐색(Search) 모드와 같은 상위 레벨의 상태를 유지합니다.

즉, 이 클래스는 **로봇의 정책(Policy)을 코드 구조 그 자체로 표현**하고 있다고 볼 수 있습니다.

<figure><img src="../.gitbook/assets/image (28).png" alt=""><figcaption></figcaption></figure>

### 4. Simulation Loop: 한 프레임의 해부

시뮬레이션이 시작되면, 매 프레임(Frame)마다 다음과 같은 순서의 작업이 반복됩니다.&#x20;

1. **외부 명령 처리:** 큐(Queue)에 쌓인 명령 확인 및 파싱
2. **물리 시뮬레이션:** MuJoCo 스텝 진행 (Physics Step)
3. **렌더링 및 관측 갱신:** 시각 정보 업데이트
4. **객체 인식(Perception):** 최신 프레임에 대한 YOLO 추론 수행
5. **종료 조건 검사:** 액션 지속 시간(Duration) 또는 목표 달성 여부 확인

### 5. Observation: 로봇은 무엇을 보는가?

이 시스템에서 로봇의 주된 관측 데이터는 단 하나입니다.

* **로봇 카메라의 최신 프레임 (`latest_frame`)**

이 프레임은 MuJoCo 렌더러로부터 직접 캡처(Capture)됩니다. 모든 시각적 인식(YOLO 등)은 오직 이 프레임만을 기준으로 수행됩니다. LiDAR와 같은 추가 센서는 보조적으로 활용될 수 있지만, 시스템의 핵심 관측은 **Vision**에 기반합니다.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-12 15-16-38.png" alt=""><figcaption></figcaption></figure>

### 6. Perception: YOLO의 통합 방식

YOLO 모델은 런타임 코드에 강하게 결합(Coupling)되어 있지 않습니다. 대신 `ObjectDetector`라는 **Perception Adapter**를 통해 느슨하게 연결됩니다.

이러한 분리 구조 덕분에 런타임은 YOLO의 내부 구현(PyTorch 버전, 모델 아키텍처 등)을 알 필요가 없으며, 비전 모델을 교체하더라도 런타임 로직은 그대로 유지될 수 있습니다.

YOLO 모듈이 반환하는 출력은 다음과 같은 **구조화된 정보**입니다.

* 객체 클래스 (Label)
* 신뢰도 (Confidence Score)
* Bounding Box 좌표
* 객체 중심 좌표 (Center Point)

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-15 11-47-29.png" alt=""><figcaption></figcaption></figure>

### 7. Perception → Action: SEARCH 로직의 구현

이 시스템의 Perception-Action Loop를 보여주는 핵심 예시는 `SEARCH_*` 액션입니다. (예: `SEARCH_HEART`, `SEARCH_SPADE`)

이 명령이 실행되면 로봇은 다음과 같은 조건부 정책(Conditional Policy)을 수행합니다.

1. **Action:** 로봇이 제자리 회전을 시작합니다.
2. **Perception:** 매 프레임 YOLO를 통해 타겟 객체(예: Heart)의 존재를 감시합니다.
3. **Decision:** 목표 객체가 감지되면(Detection), 즉시 회전을 멈추고 **정지(STOP)** 합니다.

이는 복잡한 강화학습이나 유한 상태 기계 없이도, **인식과 행동을 실시간으로 연결**한 구조적 구현의 예시입니다.



### 8. Action: 로봇의 제어 방식

본 프로젝트에서는 연속적인 제어 값(Continuous Control) 대신, 의미 기반의 문자열 명령(Semantic String Command)을 사용합니다.

* **명령 예시:** `"FORWARD"`, `"LEFT"`, `"STOP"`

각 명령은 내부적으로 좌/우 바퀴의 각속도로 변환되며, **미리 정의된 지속 시간(Duration)** 동안 수행된 후 자동으로 종료됩니다. 이러한 방식은 다음과 같은 장점이 있습니다.

* **안전성:** 명령이 무한히 지속되지 않으므로 로봇의 폭주를 방지합니다.
* **디버깅 용이성:** 사람이 읽을 수 있는 명령어로 제어되므로 로그 분석이 쉽습니다.
* **LLM 연동성:** 언어 모델이 생성하기 쉬운 형태의 출력입니다.

<figure><img src="../.gitbook/assets/image (29).png" alt=""><figcaption></figcaption></figure>

### 9. Reasoning: LLM의 역할

이 런타임 자체는 고차원적인 추론을 수행하지 않습니다. 추론은 외부 에이전트인 LLM에게 위임됩니다.

LLM은 다음과 같은 정보를 입력받아 "무엇을 해야 하는지"를 결정합니다.

* 현재 YOLO 탐지 결과 (Context)
* 사용자의 자연어 명령 (Instruction)

LLM이 결정한 결과는 문자열 명령 형태로 `command_queue`에 삽입됩니다. 런타임은 이 큐를 소비하며 "어떻게 실행할지"를 책임집니다. 즉, **두뇌(LLM)와 신체(Runtime)의 역할이 명확히 분리**되어 있습니다.

> **\[그림] LLM ↔ Runtime 상호작용**
>
> _(LLM이 Command Queue로 명령을 보내고, Runtime이 이를 실행하는 구조)_

### 10. 결론: VLA 아키텍처와의 대응

이 시뮬레이션 구조는 VLA 모델의 핵심 구성 요소들을 모두 포함하고 있습니다.

* **Vision:** 로봇 카메라 + YOLO Adapter
* **Language:** 문자열 기반 명령 체계, LLM 연동
* **Action:** 물리 엔진 기반의 구동기 제어
* **World:** MuJoCo 물리 시뮬레이션
* **Policy:** 명시적인 런타임 실행 구조

유일한 차이점은 이 모든 것이 신경망 학습이 아닌, **소프트웨어 아키텍처로 구현**되었다는 점입니다.

> **\[그림] VLA 대응 관계도**
>
> _(현재 시스템의 각 모듈이 VLA의 Vision, Language, Action 컴포넌트와 어떻게 매핑되는지 보여주는 그림)_

#### 📚 다음으로 읽으면 좋은 문서

* **\[MuJoCo: API Reference]**: 위에서 설명한 개념들이 실제 코드로 어떻게 구현되었는지 확인합니다.

이 문서는 코드를 줄 단위로 설명하기보다는, **시스템이 생각하고 작동하는 방식**을 설명하는 데 초점을 맞췄습니다. 이제 API 문서를 보시면, 각 함수가 전체 흐름 속에서 어떤 위치를 차지하는지 명확히 보일 것입니다.
