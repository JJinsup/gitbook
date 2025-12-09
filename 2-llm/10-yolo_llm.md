---
description: YOLO모델로 이미지 속 객체를 탐지하고, 그 정보를 Gemini API에 전달하여 복합적인 상황 추론을 수행하는 애플리케이션을 다룹니다.
---

# 📚 \[10] YOLO & GEMINI

### 0. YOLO(You Only Look Once)란?

{% embed url="https://docs.ultralytics.com/ko/" %}

프로젝트를 시작하기 전, 우리의 '눈'이 되어줄 YOLO 모델에 대해 자세히 알아봅니다.

#### 0.1 YOLO의 기원과 혁신 (Origins)

YOLO(You Only Look Once)는 2016년 Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi가 CVPR에서 발표한 획기적인 컴퓨터 비전 모델입니다. 발표 당시 OpenCV의 People's Choice Awards를 수상하며 큰 주목을 받았습니다.

* **배경**: YOLO 이전에는 R-CNN과 같은 모델들이 주로 사용되었으나, 속도가 느려 실시간 애플리케이션에는 적합하지 않았습니다.
* **핵심 아이디어**: YOLO는 객체 탐지를 **회귀(Regression) 문제**로 재정의했습니다. 즉, 이미지 픽셀에서 바로 바운딩 박스(Bounding Box) 좌표와 클래스 확률(Class Probability)을 예측하는 방식입니다.
* **One-Stage Detector**: 기존의 모델들이 '객체가 있을 만한 곳을 찾는 단계(Region Proposal)'와 '그 객체가 무엇인지 분류하는 단계(Classification)'를 나누어 처리했던 것과 달리, YOLO는 이 모든 과정을 하나의 신경망(End-to-End Differentiable Network)에서 한 번에 처리합니다. 이 덕분에 압도적인 속도를 자랑합니다.

#### 0.2 YOLO의 진화 (Evolution)

초기 YOLO(v1)는 Joseph Redmon이 개발한 **Darknet**이라는 유연한 프레임워크 위에서 탄생했습니다. 이후 수많은 연구자들에 의해 발전하며 다양한 버전이 출시되었습니다.

* **초기 (v1\~v3)**: Joseph Redmon이 주도하여 개발했으며, 실시간 객체 탐지의 가능성을 열었습니다.
* **중기 이후 (v4\~)**: 다른 연구자들에 의해 다양한 버전이 출시되었습니다. 각 버전은 정확도 향상, 경량화, 새로운 아키텍처 도입 등 저마다의 목표를 가지고 발전해 왔습니다.
* **최신 (YOLOv11-)**: 이 프로젝트에서 사용하는 **YOLOv11**은 Ultralytics사에서 개발한 아키텍처입니다. 객체 탐지뿐만 아니라 세그멘테이션(Segmentation), 분류(Classification), 자세 추정(Pose Estimation), 회전된 바운딩 박스(OBB) 탐지 등 다양한 기능을 지원하며, Microsoft COCO 벤치마크에서도 뛰어난 성능(mAP)을 보여줍니다.

#### 0.3 왜 YOLO인가? (Why Use YOLO?)

YOLO는 다음과 같은 이유로 수많은 엔지니어들에게 사랑받고 있습니다.

1. **압도적인 속도 (Speed)**: 높은 FPS(초당 프레임 수)로 비디오 피드를 처리할 수 있어, 빠르게 움직이는 물체 추적이나 실시간 모니터링에 최적화되어 있습니다.
2. **높은 정확도 (Accuracy)**: MS COCO 데이터셋 등에서 입증된 최신 기술(State-of-the-art) 수준의 정확도를 자랑합니다.
3. **활발한 커뮤니티 (Open Source)**: 오픈 소스로 공개되어 있어 방대한 자료와 커뮤니티 지원을 받을 수 있습니다.

이러한 장점들 덕분에 공장 침입자 감지, 건설 현장 차량 모니터링, 도로 교통량 분석, 산불 연기 감지, 작업자 안전 장비 착용 확인 등 다양한 분야에서 활용되고 있습니다.

### 1. 프로젝트 개요

단순히 물체를 찾는 것을 넘어, 그 물체들이 어떤 관계에 있고 전체적인 상황이 어떠한지 AI에게 물어보는 프로젝트입니다.

* **YOLO (v11)**: 이미지에서 사람, 강아지, 자동차 등의 위치와 종류를 빠르고 정확하게 찾아냅니다. (The Eye)
* **Gemini (Pro/Flash)**: YOLO가 찾은 정보를 바탕으로 "이게 무슨 상황이야?", "강아지는 어디 있어?" 같은 질문에 답합니다. (The Brain)

### 2. 환경 설정 (Setup)

#### 2.1 필수 라이브러리 설치

이미지 처리와 AI 모델 구동을 위해 다음 패키지들을 설치합니다.

```shellscript
pip install opencv-python ultralytics google-genai python-dotenv pyyaml ipython pillow
```

#### 2.2 프로젝트 구조

다음과 같은 폴더 구조를 권장합니다.

```shellscript
my-project/
├── .env                # API 키 저장
└── learn_LLM           # 메인 코드 폴더
    └── src/
        ├── prompt.yaml # AI 성격/지침 설정 파일
        └── dogman.jpg  # 테스트용 이미지
```

#### 2.3 설정 파일 준비

**1) `.env` 파일**

```shellscript
GEMINI_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**2) `src/prompt.yaml` 파일** AI의 행동 지침(System Instruction)을 별도 파일로 관리하면 유지보수가 쉽습니다.

```yaml
template: |
  당신은 CCTV 관제 시스템의 AI 분석관입니다.
  제공된 JSON 데이터는 화면에 감지된 객체들의 목록과 좌표입니다.
  이 정보를 바탕으로 현재 상황을 묘사하고, 보안상 특이점이 있는지 분석하세요.
  답변은 간결하고 전문적인 톤으로 작성하세요.
```

### 3. 핵심 코드 구현

#### 3.1 라이브러리 및 설정 로드

환경 변수와 YAML 프롬프트를 불러옵니다.

```python
import cv2
import json
import os
import yaml
from google import genai
from google.genai import types
from ultralytics import YOLO
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
```

YAML 파일에서 시스템 프롬프트 불러오기

```python
yaml_path = "src/prompt.yaml"

try:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        # yaml 파일의 'template' 키에 해당하는 내용을 가져옵니다.
        SYSTEM_INSTRUCTION = config_data["template"]
        print(f"'{yaml_path}'에서 프롬프트를 성공적으로 로드했습니다.")
except FileNotFoundError:
    print(f"'{yaml_path}' 파일을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
    SYSTEM_INSTRUCTION = "당신은 AI 어시스턴트입니다."
```

#### 3.2 YOLO로 객체 탐지 (The Eye)

이미지를 불러와 YOLO 모델로 분석하고, 감지된 객체 정보를 JSON으로 변환하는 과정을 단계별로 살펴봅니다.

**3.2.1. 이미지 로드 및 리사이즈**

가장 먼저 분석할 이미지를 불러옵니다. 파일 경로가 잘못되었거나 이미지가 없을 경우를 대비해 명확한 에러 메시지를 띄웁니다.

```python
img_path = "src/dogman.jpg" # 다운로드 받은 이미지 파일명

if not os.path.exists(img_path):
    print("이미지 파일이 없습니다. 경로를 확인해주세요.")
    # 테스트용 빈 캔버스
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
else:
    frame = cv2.imread(img_path)

frame = cv2.resize(frame, (1600, 900))
```

**3.2.2. YOLO 모델 실행**

YOLOv11 모델(`yolo11n.pt`)을 로드하고 이미지를 넣어 추론(Inference)을 시작합니다. `conf=0.3`은 "30% 이상 확신하는 것만 찾아라"라는 의미입니다.

```python
# YOLO 실행 (최신 v11 모델 사용)
model = YOLO("yolo11n.pt") 
results = model(source=frame, conf=0.3, verbose=False)
```

**3.2.3. 데이터 추출 준비**

감지된 객체들의 정보를 담을 빈 딕셔너리(`info`)와, 박스를 그려 넣을 이미지 사본(`annot_frame`)을 준비합니다.

```python
# 감지된 정보 추출 및 시각화 준비
# 정보 추출
info = {}
annot_frame = frame.copy()

# 감지된 목록 확인용 리스트
detected_labels = []
```

**3.2.4. 감지 결과 반복 처리 (핵심)**

YOLO는 한 번에 여러 객체를 찾기 때문에 반복문을 돕니다. 여기서 각 객체의 **클래스(이름)**, **신뢰도(점수)**, **위치(좌표)**&#xB97C; 추출합니다.

```python
# 감지 결과 반복문 처리
for res in results:
    for b in res.boxes:
        # (1) 클래스 ID 및 이름 추출
        cls = int(b.cls[0].item())
        label = res.names.get(cls, str(cls)) # 예: 'person', 'dog'
        
        # (2) 신뢰도(Confidence) 추출 (소수점 둘째 자리까지)
        conf = round(b.conf[0].item(), 2)
        
        # (3) 바운딩 박스 좌표 추출 (좌상단 x1,y1 / 우하단 x2,y2)
        x1, y1, x2, y2 = map(int, b.xyxy[0]) 
        x, y, w, h = map(int, b.xywh[0])
        
        # 감지된 것 기록
        detected_labels.append(label)
```

**3.2.5. 정보 구조화 (setdefault 활용)**

같은 종류의 객체가 여러 개일 수 있습니다(예: 강아지 2마리). 이때 `setdefault`를 사용하면 코드를 간결하게 짤 수 있습니다.

* **설명**: `info` 딕셔너리에 `label`(예: 'dog')이라는 키가 **없으면**, 빈 리스트 `[]`를 값으로 넣어줍니다. 그리고 그 리스트를 반환합니다.
* **추가**: 반환된 리스트(비었거나 이미 데이터가 있는 리스트)에 `.append()`를 사용하여 현재 감지된 객체의 정보를 추가합니다.
* **결과**: `{'dog': [{'bbox':...}, {'bbox':...}], 'person': [...]}` 형태로 데이터가 예쁘게 정리됩니다.

```python
        # (4) 정보 저장
        info.setdefault(label, []).append({
            "location": [x, y],
            "size": w * h,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })
```

**3.2.6. 시각화 및 JSON 변환**

분석이 끝난 객체 위에 초록색 박스와 글씨를 쓰고, Gemini에게 전달하기 위해 파이썬 딕셔너리를 문자열(JSON) 형태로 변환합니다.

```python
        # (5) 시각화 (이미지에 박스와 글자 그리기)
        cv2.rectangle(annot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annot_frame, f"{label} {conf}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 4. JSON 문자열로 변환 (Gemini에게 줄 데이터)
detected_info_json = json.dumps(info, ensure_ascii=False, indent=2)
```

**3.2.7. YOLO가 감지한 물체 목록 확인**

```python
print(f"YOLO가 감지한 물체 목록: {list(set(detected_labels))}")
display(Image.fromarray(cv2.cvtColor(annot_frame, cv2.COLOR_BGR2RGB))) 
```

> **`YOLO가 감지한 물체 목록: ['person', 'car', 'dog']`**

<figure><img src="../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

#### 3.3 Gemini에게 물어보기 (The Brain)

YOLO가 추출한 `detected_info_json`을 프롬프트에 포함시켜 Gemini에게 질문합니다.

```python
def ask_gemini(question, detected_info):
    # 프롬프트 엔지니어링: 객체 정보 + 사용자 질문 결합
    user_content = f"""
    # 감지된 객체 정보 (JSON):
    {detected_info}

    # 질문:
    {question}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION, # YAML에서 불러온 페르소나 적용
                temperature=0.1, # 사실 기반 답변을 위해 낮음 설정
            ),
            contents=user_content
        )
        return response.text
    except Exception as e:
        return f"에러 발생: {e}"

# 실행 예시
question = "지금 보이는 상황을 요약해줘."
answer = ask_gemini(question, detected_info_json)
print(f"🤖 AI 분석: {answer}")
```

**3.3.1. GEMINI 호출 함수**

```python
def ask_gemini(question, detected_info):
    # 사용자 질문 구성
    user_content = f"""
    # 감지된 객체 정보 (JSON):
    {detected_info}

    # 질문:
    {question}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            config=types.GenerateContentConfig(
                # 여기서 YAML에서 불러온 시스템 지침을 적용합니다!
                system_instruction=SYSTEM_INSTRUCTION, 
                temperature=0.1,
            ),
            contents=user_content
        )
        return response.text
    except Exception as e:
        return f"에러 발생: {e}"
```

**3.3.2. 테스트 실행**

```python
# 테스트 실행
print("\n[Gemini 분석 결과]")
print(f"Q: 지금 보이는 상황을 설명해줘")
print(f"A: {ask_gemini('지금 보이는 상황을 설명해줘', info_str)}")
```

> **`[Gemini 분석 결과] Q: 지금 보이는 상황을 설명해줘 A: 이미지 중앙에서 약간 오른쪽 아래에 사람이 한 명 있습니다. 이 사람은 이미지의 상당 부분을 차지하고 있으며, 신뢰도는 0.9입니다.`**
>
> **`이미지 중앙 부분에는 총 네 대의 자동차가 감지되었습니다.`**
>
> **`그 오른쪽으로 두 대의 자동차가 더 있으며, 이 중 한 대는 중앙에, 다른 한 대는 중앙에서 약간 오른쪽으로 치우쳐 있습니다.`**\
> **`가장 왼쪽에 있는 두 대의 자동차는 중앙에서 약간 왼쪽으로 치우쳐 있으며, 서로 가까이 붙어 있습니다.`**
>
> **`이미지 왼쪽 하단에는 개 한 마리가 있습니다.`**

### 💡 4. 응용 팁

1. **멀티모달 확장**: 위 예제는 JSON 텍스트만 넘겼지만, `contents`에 실제 이미지(`frame`)를 함께 넘기면 Gemini가 시각 정보까지 더해 더욱 정교한 답변을 할 수 있습니다.
2. **YAML 프롬프트 튜닝**: `src/prompt.yaml`의 내용을 바꿔보세요. "시인처럼 묘사해줘"라고 적으면 감성적인 글을 써줍니다.
3. **실시간 처리**: 이 코드를 루프에 넣고 웹캠(`cv2.VideoCapture(0)`)과 연결하면 실시간 AI 영상 분석기가 됩니다.
