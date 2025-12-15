---
description: >-
  MuJoCo 시뮬레이션 환경에 Ultralytics YOLO를 통합하는 ObjectDetector 클래스를 분석하고, 실제 시뮬레이션에서
  YOLO가 정상적으로 동작하는지 확인하는 테스트 과정을 다룹니다.
---

# \[18] MuJoCo: YOLO 성능 테스트

<figure><img src="../.gitbook/assets/4 (1).jpg" alt=""><figcaption></figcaption></figure>

### 1. ObjectDetector 코드 구조 및 설계

`ObjectDetector`는 Ultralytics YOLO를 MuJoCo 시뮬레이션 파이프라인에서 안정적으로 사용하기 위해 설계된 **래퍼(Wrapper) 클래스**입니다.

이 클래스의 핵심 목적은 다음과 같습니다.

1. **의존성 격리:** YOLO 라이브러리 의존성을 `utils` 레벨에 가두어 메인 런타임의 복잡도를 낮춥니다.
2. **데이터 구조화:** 추론 결과를 후처리(제어, LLM 입력)하기 쉬운 Dictionary 형태로 변환합니다.
3. **스레드 안전성:** 시뮬레이션 루프와 추론 루프 간의 충돌을 방지합니다.

#### 1.1 라이브러리 및 의존성 처리

```
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    _YOLO_AVAILABLE = False
```

* **방어적 설계:** `ultralytics`가 설치되지 않은 환경(예: 렌더링 전용 서버, 경량 테스트 환경)에서도 파일 `import` 자체가 깨지지 않도록 처리했습니다.
* **플래그 관리:** 실제 YOLO 사용 가능 여부는 `_YOLO_AVAILABLE` 플래그로 관리하며, 객체 초기화 시 명시적으로 에러를 발생시켜 사용자에게 알립니다.

#### 1.2 데이터 구조

```
@dataclass
class DetectedBox:
    confidence: float
    bbox: list
    center: list
```

* 단일 Bounding Box를 표현하기 위한 데이터 구조입니다.
* 현재는 유연성을 위해 Dictionary 출력을 주로 사용하지만, 추후 타입 안정성(Type Safety)이나 로직 확장이 필요할 때를 대비해 정의된 구조입니다.

#### 1.3 ObjectDetector 클래스 상세

이 클래스는 단순한 YOLO 호출을 넘어, **MuJoCo ↔ OpenCV ↔ YOLO** 사이를 연결하는 어댑터 역할을 수행합니다.

**초기화 및 Thread-Safe 설계**

```
def __init__(self, weight_path: str, conf: float = 0.5):
    self.model = YOLO(weight_path)
    self._lock = threading.Lock()  # 핵심
```

* **`_lock`:** 시뮬레이션 렌더링 루프와 LLM/제어 스레드가 동시에 추론을 요청할 경우를 대비해 `threading.Lock`을 사용하여 동시성 문제를 예방합니다.

**핵심 메소드: `detect_dict`**

제어 로직이나 LLM이 즉시 사용할 수 있도록 **구조화된 탐지 결과**를 반환하는 가장 중요한 함수입니다.

* **입력:** `frame_bgr` (OpenCV 기준 BGR 이미지, MuJoCo 카메라 캡처본)
* **출력:** 클래스별로 그룹화된 객체 정보 Dictionary

**출력 데이터 예시:**

```
{
  "heart": [
    {
      "confidence": 0.92,
      "bbox": [100, 50, 200, 150],
      "center": [150, 100]
    }
  ],
  "diamond": []
}
```

> **설계 의도:** 단순히 박스 좌표만 넘기는 것이 아니라 Label 기준으로 데이터를 그룹핑함으로써, 탐색 로직(Search Logic)이나 **LLM 프롬프트 생성** 단계에서 별도의 가공 없이 바로 데이터를 사용할 수 있게 합니다.

**시각화 메소드: `detect_image`**

```
def detect_image(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    # 내부적으로 result.plot() 사용
    return plotted_image
```

* YOLO 추론 결과(Bounding Box, Label)가 그려진 이미지를 반환합니다.
* 주로 `cv2.imshow()`를 통한 **실시간 디버깅**이나 **데모 영상 기록** 용도로 사용됩니다.

### 2. 실습: MuJoCo-YOLO 성능 테스트

이 단계에서는 앞서 설명한 `ObjectDetector`가 MuJoCo 시뮬레이션과 결합되어 정상적으로 동작하는지 확인합니다.

#### 2.1 테스트 목적

1. **이미지 변환:** MuJoCo 로봇 카메라 → OpenCV 이미지 변환이 정상적인가?
2. **모델 로드:** `.pt` 가중치 파일이 정상적으로 로드되는가?
3. **시각적 확인:** Bounding Box와 Confidence Score가 올바른 위치에 표시되는가?

#### 2.2 실행 코드

아래 코드는 YOLO가 통합된 `TurtlebotFactorySim`을 실행하는 최소 단위 예제입니다.

```
import os
import sys

# 1. 프로젝트 루트 경로 설정 (환경에 맞게 수정 필요)
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

# 2. 통합 시뮬레이션 클래스 Import
from tb3_sim import TurtlebotFactorySim

# 3. 리소스 경로 설정
XML_PATH = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_cards.xml")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "scripts", "best.pt") # 학습된 모델 경로

print(f"XML Path: {XML_PATH}")
print(f"Weights: {YOLO_WEIGHTS}")

if __name__ == "__main__":
    # 시뮬레이션 인스턴스 생성
    sim = TurtlebotFactorySim(
        xml_path=XML_PATH,
        use_yolo=True,              # 핵심: YOLO 활성화
        yolo_weight_path=YOLO_WEIGHTS,
        yolo_conf=0.5,              # Confidence Threshold
    )

    # 시뮬레이션 시작
    # 내부 동작: MuJoCo Step -> 렌더링 -> 캡처 -> YOLO 추론 -> 시각화
    sim.start()
```

#### 2.3 실행 결과 확인

코드를 실행하면 다음과 같은 3개의 창이 나타나야 정상입니다.

1. **Observer View:** 전체 시뮬레이션 환경을 조망하는 MuJoCo 메인 뷰
2. **Camera View (Robot):** 터틀봇에 부착된 카메라의 Raw 시점
3. **Robot YOLO View (OpenCV):** YOLO Bounding Box가 오버레이 된 결과 화면

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-15 11-47-29 (1).png" alt=""><figcaption></figcaption></figure>

**성공 기준:**

* `Robot YOLO View` 창에서 카드 객체 위에 **Bounding Box**와 **Confidence Score**가 실시간으로 표시된다면 성공입니다.
* 종료하려면 OpenCV 창에서 `q`를 누르거나, MuJoCo 메인 창을 닫습니다.

#### 2.4 요약 및 다음 단계

여기까지 진행하셨다면 **\[MuJoCo] → \[Camera] → \[OpenCV] → \[YOLO]** 로 이어지는 데이터 파이프라인이 완성된 것입니다. 단순히 화면에 박스를 그리는 것을 넘어, 이제 이 인식 정보를 로봇의 지능으로 연결할 차례입니다.

**Next Steps:**

* 👉 `detect_dict()` 결과를 활용한 **탐색(Search) 알고리즘** 구현
* 👉 인식된 객체 정보를 JSON으로 변환하여 **LLM(Gemini)에 상황 설명** 전달
* 👉 LLM의 판단에 따른 **로봇 행동(Action) 제어**
