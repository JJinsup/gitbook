---
description: >-
  PDF 문서 하나만 있으면, LLM 학습(Fine-tuning)에 필요한 고품질의 질문-답변(Q&A) 데이터셋을 자동으로 생성하고
  증강(Augmentation)하는 파이프라인을 구축하는 방법을 다룹니다.
---

# 📚 \[13] 데이터셋 생성 + 증강

{% embed url="https://www.ibm.com/kr-ko/think/topics/data-augmentation" %}

_**PDF 출처:**_ [https://www.6gforum.or.kr/html/sub/member/view.php?bo\_table=gallery\&wr\_id=191](https://www.6gforum.or.kr/html/sub/member/view.php?bo_table=gallery\&wr_id=191)

### 0. 데이터 증강(Data Augmentation)이란?

프로젝트를 시작하기에 앞서, 우리가 수행하려는 '데이터 증강'이 무엇인지 간단히 짚고 넘어갑니다.

#### 0.1 정의 및 필요성

**데이터 증강**은 기존 데이터를 변형하여 **새로운 데이터 샘플을 인위적으로 생성**하는 기술입니다. 머신러닝의 핵심 원칙 중 하나는 "크고 다양한 데이터셋이 모델 성능을 높인다"는 것인데, 현실에서는 윤리적 문제나 비용 등의 이유로 데이터를 충분히 확보하기 어려운 경우가 많습니다.

* **목적**: 데이터의 양과 다양성을 늘려 **불완전한 데이터셋을 보완**하고, 모델의 **과적합(Overfitting)을 방지**하며 일반화 성능을 개선합니다.
* **합성 데이터와의 차이**: 아예 없는 데이터를 생성하는 '합성 데이터(Synthetic Data)'와 달리, 증강은 **기존 데이터를 기반으로 변형**한다는 점에서 차이가 있습니다.

#### 0.2 텍스트 증강 기술

이미지뿐만 아니라 텍스트(NLP) 분야에서도 다양한 증강 기법이 사용됩니다.

* **규칙 기반 (Rule-based)**: 단어의 무작위 삭제/삽입, 동의어 교체(Synonym Replacement), 문장 구조 변경(능동태 ↔ 수동태) 등 비교적 간단한 방법입니다.

<figure><img src="../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>

*   **신경망 기반 (Neural-based)**:

    * **역번역 (Back-translation)**: 한글 → 영어 → 한글로 번역하여 의미는 같지만 표현이 다른 문장을 생성합니다.

    <figure><img src="../.gitbook/assets/image (14).png" alt=""><figcaption></figcaption></figure>

    * **생성 모델 활용**: 이 프로젝트에서 사용하는 방식으로, LLM(Gemini)에게 "의미는 유지하되 표현을 바꿔라"고 지시하여 고품질의 문장을 생성합니다.

### 1. 개요

LLM을 특정 도메인(법률, 의학, 사내 규정 등)에 맞게 파인 튜닝하려면 양질의 Q\&A 데이터가 필수적입니다. 하지만 사람이 직접 수천 개의 질문과 답변을 만드는 것은 시간과 비용이 많이 듭니다.

이 프로젝트는 **Gemini API**와 **LangChain**을 활용하여 이 과정을 자동화합니다.

#### 핵심 기능

1. **PDF 처리**: 긴 문서를 AI가 읽기 좋은 크기(Chunk)로 자동 분할합니다.
2. **Q\&A 생성 (Generation)**: 각 청크 내용을 바탕으로 Gemini가 예상 질문과 모범 답변을 생성합니다.
3. **데이터 증강 (Augmentation)**: 생성된 질문을 다양한 표현으로 바꿔(Paraphrasing), 데이터 양을 4배 이상 뻥튀기합니다.

### 2. 환경 설정 (Setup)

#### 2.1 필수 라이브러리 설치

PDF 처리와 AI 모델 연동을 위해 다음 패키지들을 설치합니다.

```bash
pip install langchain-text-splitters langchain-community tdqm pymupdf
```

#### 2.2 프로젝트 구조

다음과 같은 폴더 구조를 권장합니다. `src` 폴더에 학습시킬 PDF 파일을 넣어주세요.

```bash
my-project/
├── .env                # API 키 저장
└── learn_LLM           # 메인 코드 폴더
    └── src/
        └── 6g_ai.pdf   # 분석할 PDF 파일
```

### 3. 핵심 파이프라인

`datagen.ipynb`의 주요 기능을 단계별로 살펴봅니다.

#### 3.1 라이브러리 & API 로드

```python
import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tqdm.notebook import tqdm  # 진행바 표시용
# PDF 처리용
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

```python
# 1. 설정
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
```

#### 3.2 PDF 로드 및 청크 분할 (Preprocessing)

긴 PDF를 한 번에 처리할 수 없으므로, 의미 단위로 잘라줍니다. `RecursiveCharacterTextSplitter`를 사용하여 문맥이 끊기지 않도록 겹치는 구간(Overlap)을 두고 자릅니다.

```python
# 2. PDF 로드 및 분할 (PyMuPDFLoader 사용 - 속도/정확도/메타데이터 확보)
def load_and_split_pdf(file_path):
    print(f"Loading PDF with PyMuPDF: {file_path}...")
    
    # (1) 로드: 페이지별로 텍스트와 메타데이터(페이지번호 등)를 가져옵니다.
    loader = PyMuPDFLoader(file_path)
    docs = loader.load() 
    
    # (2) 분할: 문맥을 고려해 자르되, 페이지 정보 등은 유지합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs) # split_text가 아니라 split_documents 사용
    
    return chunks
```

#### 3.3 Q\&A 데이터 생성 (Generation)

잘라진 텍스트 조각(Chunk)을 Gemini에게 보여주고, "이 내용을 바탕으로 학습용 질문 3개를 만들어줘"라고 시킵니다. 결과는 JSON 형식으로 받아 처리하기 쉽게 만듭니다.

```python
# 3. 데이터셋 생성 루프
def generate_dataset_gemini(chunks, output_file):
    base_prompt = """
    당신은 해당 도메인의 전문가입니다. 
    아래 [Context]를 보고 학습용 질문-답변 쌍 3개를 JSON으로 만드세요.
    ...
    """
    # ... (Gemini API 호출 및 JSON 파싱 로직) ...
```

```python
# PDF 파일 경로
pdf_filename = "src/6g_ai.pdf" 

# 실행
chunks = load_and_split_pdf(pdf_filename)
generate_dataset_gemini(chunks, "6g_ai_dataset.jsonl")
```

`저장 완료: 6g_ai_dataset.jsonl (총 18개 데이터 쌍)`

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

#### 3.4 데이터 증강 (Augmentation)

생성된 Q\&A 데이터가 100개라면, 이를 400개로 늘리는 과정입니다. "답변은 그대로 두고, 질문만 다르게 표현해봐"라고 요청하여 모델이 다양한 질문 패턴을 학습할 수 있게 돕습니다.

* **원본**: "6G의 주요 특징은 무엇인가요?"
* **증강 1**: "6G 통신 기술이 기존 5G와 차별화되는 점을 설명해주세요."
* **증강 2**: "차세대 이동통신 6G가 가지는 핵심적인 기술적 특성은?"

```python
def augment_dataset_gemini(input_file, output_file):
    aug_prompt = """
    아래 주어진 질문(Instruction)과 답변(Output)을 보고, 
    **답변은 그대로 유지하되 질문의 표현만 다르게 바꾼 유사 질문 3개**를 생성하세요.
    ...
    """
    # ... (Gemini API 호출 및 데이터 추가 로직) ...
```

```python
augment_dataset_gemini("6g_ai_dataset.jsonl", "6g_ai_dataset_augmented.jsonl")
```

`증강 완료! 총 36개 데이터가 '6g_ai_dataset_augmented.jsonl'에 저장됨.`

<figure><img src="../.gitbook/assets/image (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

### 4. 결과 확인

생성된 `6g_ai_dataset_augmented.jsonl` 파일은 다음과 같은 형태를 가집니다. 이 파일은 바로 LoRA나 Fine-tuning 학습에 사용할 수 있습니다.

```json
{"instruction": "6G 네트워크의 초저지연 특성에 대해 설명하시오.", "input": "", "output": "6G는 1ms 이하의 초저지연을 목표로 하며..."}
{"instruction": "6G 기술이 실현하고자 하는 지연 시간 목표치는 얼마인가?", "input": "", "output": "6G는 1ms 이하의 초저지연을 목표로 하며..."}

```

### 5. 활용 팁

1. **프롬프트 튜닝**: `generate_dataset_gemini` 함수의 프롬프트를 수정하여 "객관식 문제를 만들어줘"나 "OX 퀴즈를 만들어줘"와 같이 다양한 형태의 데이터를 생성할 수 있습니다.
2. **온도 조절**: `temperature` 값을 높이면(0.8 이상) 더 창의적인 질문이 나오고, 낮추면(0.2 이하) 더 정직하고 딱딱한 질문이 나옵니다.
3. **대용량 처리**: PDF가 수백 페이지가 넘어가면 API 비용과 시간이 많이 들 수 있습니다. 이럴 땐 청크 중 일부만 샘플링(`random.sample`)해서 테스트해보는 것이 좋습니다.
