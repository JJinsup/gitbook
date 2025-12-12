---
description: 이미지 데이터셋 생성부터 YOLOv11 모델 파인튜닝까지 실습
---

# \[16] YOLO 파인튜닝

## MuJoCo 데이터 생성부터 학습까지

MuJoCo 시뮬레이터에서 카드 이미지를 생성(Capture)하고, Roboflow의 SAM 3 모델을 이용해 자동 라벨링을 수행한 뒤, Google Colab(T4 GPU)에서 YOLOv11 모델을 학습시키는 전체 워크플로우입니다.

<figure><img src="../.gitbook/assets/image (25).png" alt=""><figcaption></figcaption></figure>

### 1. MuJoCo에서 이미지 생성하기

#### 1.1 목적

학습에 필요한 카드 이미지 데이터셋(프레임)을 시뮬레이터에서 생성합니다. 이 단계의 핵심 목표는 **"많이, 다양하게"** 데이터를 300장 정도 확보하는 것입니다.&#x20;

* **배경/조명/카메라:** 거리와 각도, 조명을 다양하게 변화
* **위치:** 터틀봇 위치를 움직이며 촬영
* **상황:** 카드가 일부 가려지거나(Occlusion) 겹쳐진(Overlapping) 상황 포함

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

#### 1.2 실행 흐름 (권장)

1. 시뮬레이션 실행 ( `tb3_tutorial.py -> capture_cards.py`)
2. 카드가 잘 보이도록 터틀봇을 직접 이동
3. 시뮬레이션 스텝을 진행하며 일정 주기마다 화면 캡처
4. 지정된 로컬 폴더에 이미지 자동 저장

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

#### 1.3 저장 폴더 규칙

Roboflow 업로드를 용이하게 하기 위해 단일 폴더에 이미지를 모아두는 것을 권장합니다.

* `img_dataset/images/000001.png`
* `img_dataset/images/000002.png`

#### 1.4 환경 로드 및 확인 (`tb3_tutorial.py` 활용)

데이터 수집(캡처)을 시작하기 전에, **카드 환경 이 정상적으로 불러와지는지 시각적으로 확인**하고 싶다면 아래 코드를 사용하세요.&#x20;

`tb3_tutorial.py`에서 XML 경로만 변경하여 실행하는 방식입니다.

```python
import os
import sys

# 1. 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

# 2. XML 파일 경로 설정 (카드 환경으로 변경)
# 기존: "tb3_factory.xml" -> 변경: "tb3_factory_cards.xml"
xml_path = os.path.join(PROJECT_ROOT, "asset", "robotis_tb3", "tb3_factory_cards.xml")
print("Using XML:", xml_path)

from tb3_sim import TurtlebotFactorySim

if __name__ == "__main__":
    # 3. 시뮬레이터 실행 (YOLO 탐지 기능은 끄고 시뮬레이션만 확인)
    sim = TurtlebotFactorySim(xml_path=xml_path, use_yolo=False)
    sim.start()
```

#### 1.5 자동 캡처 스크립트 예시 (`capture_cards.py`)

다음은 MuJoCo 뷰어를 실행하고 10 프레임마다 자동으로 이미지를 캡처하여 저장하는 파이썬 스크립트 예시입니다.&#x20;

**필요에 따라 `XML_PATH`와 `OUT_DIR` 경로를 수정하여 사용하세요.**

```python
import os
import time
import cv2
import mujoco as mj

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))  # 프로젝트 루트로

from utils.mujoco_renderer import MuJoCoViewer

# 1. 사용할 XML 파일 경로 설정
XML_PATH = "/data/jinsup/js_mujoco/asset/robotis_tb3/tb3_factory_cards.xml"

# 2. 이미지가 저장될 위치 설정
OUT_DIR = "img_dataset/images"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    viewer = MuJoCoViewer(model, data)

    idx = 0
    frame = 0
    try:
        while not viewer.should_close():
            time_prev = data.time
            while data.time - time_prev < 1.0 / 60.0:
                viewer.step_simulation()

            viewer.render_main()
            viewer.render_robot()
            viewer.poll_events()

            # 프레임 카운트 증가
            frame += 1

            # 10프레임마다 한 번씩만 캡처하여 저장
            if frame % 10 == 0:
                img = viewer.capture_img()
                out_path = os.path.join(OUT_DIR, f"img_{idx:05d}.jpg")
                cv2.imwrite(out_path, img)
                print("saved:", out_path)
                idx += 1

            time.sleep(0.01)

    finally:
        viewer.terminate()

if __name__ == "__main__":
    main()
```

### 2. Roboflow로 자동 라벨링하기 (SAM 3)

#### 2.1 새 프로젝트 생성

1. **Roboflow** 접속 및 로그인
2. **Create New Project** 클릭
3. **Project type:** Object Detection
4. **Annotation type:** Traditional
5. 프로젝트 생성 완료

<figure><img src="../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/image (26).png" alt=""><figcaption></figcaption></figure>

#### 2.2 이미지 업로드

1. 생성한 프로젝트 내부로 진입
2. **Upload** 탭에서 앞서 생성한 `img_dataset/images` 폴더의 이미지들을 업로드

#### 2.3 SAM 3 자동 라벨링 설정

Roboflow의 **Auto Label (Masks/SAM 3)** 기능을 사용합니다. 모델이 카드의 심볼을 인식하여 박스를 치도록 클래스와 설명을 등록합니다.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-29-51.png" alt=""><figcaption></figcaption></figure>

**📋 SAM 3 입력용 프롬프트 (4개 클래스)** 아래 내용을 하나씩 입력하고 `+ Add Class`를 눌러 총 4개를 등록합니다.

| 클래스 (Class)  | 이름 (Class name) | 시각적 설명 (Visual description)                |
| ------------ | --------------- | ------------------------------------------ |
| **♣️ 클로버**   | `club`          | ace playing card with a black club symbol  |
| **♦️ 다이아몬드** | `diamond`       | ace playing card with a red diamond symbol |
| **♥️ 하트**    | `heart`         | ace playing card with a red heart symbol   |
| **♠️ 스페이드**  | `spade`         | ace playing card with a black spade symbol |

#### 2.4 품질 확인 및 적용

1. **Generate Test Results**를 클릭하여 샘플 이미지에 대한 라벨링 결과를 확인합니다.
   * _체크 포인트:_ 심볼에 박스가 정확히 잡히는가? 배경을 오인식하지 않는가?
2. 결과가 양호하다면 **Auto Label With This Model**을 클릭하여 전체 이미지 라벨링을 수행합니다.

#### 2.5 검수 및 데이터셋 확정

1. 자동 라벨링 완료 후 검수 화면으로 이동
2. 오탐(False Positive)이나 미탐(False Negative)을 빠르게 수정
3. **Approved** 버튼 클릭
4. **Add Approved to Dataset**을 눌러 학습 가능한 데이터셋(Train/Valid/Test 분할)으로 확정 -> _**Train : Valid 7:3 추천**_

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-35-32.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-40-03.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-42-25.png" alt=""><figcaption></figcaption></figure>

<p align="center">ㄷ</p>

### 3. Roboflow에서 YOLOv11용 버전 생성

#### 3.1 버전 생성 (Create New Version)

좌측 사이드바의 **Versions** 탭으로 이동하여 **Create New Version**을 클릭합니다.

#### 3.2 전처리 및 증강 설정 (권장값)

기본적인 증강을 적용하여 모델의 강건성을 높입니다.

* **Preprocessing:** Auto-Orient (기본값), Resize X

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-44-55.png" alt=""><figcaption></figcaption></figure>

*   **Augmentation:**

    * **Flip:** Horizontal, Vertical
    * **Rotation:** ±10도

    <figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-58-34.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 20-58-39.png" alt=""><figcaption></figcaption></figure>

* **생성:** `Create` (예: 673 images -> 3x 증강)

#### 3.3 데이터셋 다운로드

버전 생성이 완료되면 우측 상단의 **Export Dataset**을 클릭하고 포맷을 선택합니다.

* **Format:** `YOLOv11`
* 다운로드된 zip 파일을 준비합니다.

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-11 21-00-20.png" alt=""><figcaption></figcaption></figure>

### 4. Google Colab에서 YOLOv11 학습시키기

#### 4.1 환경 설정

1. **Google Colab** 접속 후 `새 노트북` 생성
2. 상단 메뉴: **런타임** → **런타임 유형 변경**
3. **하드웨어 가속기:** `T4 GPU` 선택

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

#### 4.2 데이터셋 업로드 및 마운트

1. Roboflow에서 다운로드한 zip 파일 압축해제하여 구글 드라이브(`MyDrive/img_dataset/`)에 업로드합니다.
2. Colab에서 드라이브를 마운트합니다.
3.  ultralytics 라이브러리를 설치합니

    ```python
    /content/drive/MyDrive/YOLO_Result
    ```

    .

    ```python
    /content/drive/MyDrive/YOLO_Result
    ```

```
# 1. 설치 및 드라이브 연결
!pip install ultralytics -U
from google.colab import drive
drive.mount('/content/drive')
```

#### 4.3 데이터셋 폴더로 이동

```
# 2. 데이터셋 폴더로 이동
%cd /content/drive/MyDrive/img_dataset
```

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

#### 4.4 학습 실행 (Training)

`data.yaml` 파일의 경로를 지정하고 학습을 시작합니다.

```
# 3. 학습 시작
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='data.yaml',      
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolo11_cards',
    project='/content/drive/MyDrive/YOLO_Result' # 결과 저장
)
```

#### 4.6 결과 확인

학습이 완료되면 `/` 경로에 가중치와 결과 그래프가 저장됩니다.

* `weights/best.pt`: 최고 성능 모델 가중치 **(다운로드 필수)**
* `results.png`: 손실(Loss) 및 성능(mAP) 그래프
* `confusion_matrix.png`: 혼동 행렬

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

학습이 성공적으로 완료되었다면, 이제 실시간 환경에 적용할 차례입니다.

1. 구글드라이브에서 `best.pt` 파일을 scripts폴더로 다운로드합니다.
2. `YOLO_Result`폴더에서 결과물도 확인해보세요.

### ✅ 트러블슈팅 체크리스트

실전에서 자주 막히는 포인트들을 점검하세요.

* **클래스 이름 일치:** Roboflow 설정 이름(`club`, `heart`...)과 코드 내 이름이 동일한가?
* **검수(Approved):** Auto Label 결과 중 엉뚱한 박스는 제거했는가?
* **경로 확인:** Colab에서 `data.yaml`의 경로가 실제 파일 위치와 일치하는가?
* **성능 문제:** 학습은 되는데 탐지율이 낮다면?
  * Roboflow에서 **증강(Augmentation)** 강도를 높여 다시 버전 생성
  * 학습 시 **Epochs**를 100 이상으로 증가
  * MuJoCo 캡처 단계에서 더 다양한 각도와 조명 데이터를 추가
