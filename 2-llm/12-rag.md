---
description: Python 라이브러리를 활용하여 문서를 처리하고 Ollama 모델과 연동하는 RAG 워크플로우 예시입니다.
---

# \[12] Ollama RAG 실습

## 1. Ollama Python (RAG) 실습

**사전 준비**

```shellscript
pip install langchain-ollama pypdf
```

#### 전체 순서 (Workflow)

_**1단계: PDF 텍스트 추출 (Document Loading)**_

* **내용**: PDF 파일 원본에서 텍스트 데이터를 읽어옵니다.
* **도구**: Python 라이브러리 `pypdf` 등을 사용합니다.

_**2단계: 청킹 (Chunking)**_

* **내용**: 추출한 긴 텍스트를 LLM이 처리하기 적합한 크기(예: 문단 단위, 500자 내외)로 자릅니다.
* **목적**:
  * 검색의 정확도 향상 (관련성 높은 부분만 찾기 위해)
  * 임베딩 효율성 증대
  * LLM의 컨텍스트 윈도우(입력 제한) 준수

_**3단계: 문서 임베딩 (Indexing)**_

* **내용**: 잘린 텍스트 조각(Chunk)들을 임베딩 모델을 통해 컴퓨터가 이해할 수 있는 벡터(숫자 목록)로 변환합니다.
* **저장**: 변환된 벡터를 벡터 저장소(Vector Store)에 저장하여 검색 가능한 상태로 만듭니다.
* **예시 모델**: `text-embedding-004`, `huggingface-embeddings` 등

_**4단계: 질문 임베딩 및 검색 (Retrieval)**_

* **내용**: 사용자의 자연어 질문(예: "이 문서 요약해줘")도 문서와 동일한 방식으로 벡터화합니다.
* **매칭**: 저장된 문서 벡터들 중 질문 벡터와 가장 유사도가 높은(거리가 가까운) 내용을 찾아냅니다.

_**5단계: 답변 생성 (Generation)**_

* **내용**: 검색 단계에서 찾아낸 **관련 내용(Context)**&#xACFC; **사용자의 질문(Query)**&#xC744; 합쳐 프롬프트를 구성합니다.
* **생성**: 이를 LLM에게 보내 문맥에 기반한 최종 답변을 생성합니다.
* **예시 모델**: `gemini-2.5-flash-lite`, `qwen3:0.6b` (Ollama 구동) 등

### 3. 실습 1: LLM을 함수처럼 사용하기 (Structured Output)

이 실습에서는 **Ollama**로 구동되는 `qwen3:0.6b` 모델을 사용하여, 비정형 자연어 텍스트에서 데이터를 추출해 정형화된 JSON 형태로 변환합니다.

#### 3.1 라이브러리 임포트 및 모델 설정

먼저 필요한 도구들을 불러오고, 로컬에 설치된 Ollama 모델을 연결합니다.

```
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 모델 초기화
# model="qwen3:0.6b": Ollama에 설치된 모델 이름을 정확히 입력합니다.
# temperature=0: 창의성을 0으로 낮추어, 항상 일관되고 사실적인 답변만 하도록 설정합니다. (데이터 추출에 필수)
llm = ChatOllama(model="qwen3:0.6b", temperature=0)
```

#### 3.2 프롬프트 템플릿 작성 (Few-shot Learning)

작은 모델일수록 구체적인 예시를 보여주는 것이 중요합니다. 이를 **Few-shot Prompting**이라고 합니다. 모델에게 "입력이 A일 때 출력은 B여야 해"라고 가르쳐줍니다.

```
# 프롬프트 템플릿 정의
template = """
You are a precise Data Extraction Module.
Extract hardware status from the input text into a JSON List format.

Examples:
Input: "Battery is 12V and Camera is OK."
Output: [
    {{"component": "Battery", "status": "Normal", "value": 12}},
    {{"component": "Camera", "status": "OK", "value": null}}
]

Input: {input}
Output:
"""

prompt = ChatPromptTemplate.from_template(template)
```

#### 3.3 체인 생성 및 실행

LangChain의 파이프(`|`) 연산자를 사용하여 `프롬프트 -> 모델 -> 출력 파서` 순서로 연결합니다.

```
# 체인(Chain) 연결
# 1. prompt: 사용자의 입력을 템플릿에 채움
# 2. llm: 채워진 프롬프트를 모델에 전달하여 답변 생성
# 3. StrOutputParser: 모델의 답변 객체에서 텍스트 부분만 깔끔하게 추출
chain = prompt | llm | StrOutputParser()

# 테스트 입력 데이터
user_input = "LiDAR 센서-> 정상 작동 중, 모터 온도가 45도라서 경고 상태야."

# 체인 실행 (Invoke)
print(f"입력: {user_input}")
response = chain.invoke({"input": user_input})

print("출력 결과:")
print(response)
```

> _**실행결과:**_\
> **\[**\
> **{"component": "LiDAR", "status": "Normal", "value": null},**\
> **{"component": "Motor", "status": "Warning", "value": 45}**\
> **]**

### 4. 실습 2: 문서 기반 전문가 (On-Device RAG)

이 실습에서는 PDF 문서를 읽어 벡터화(Embedding)하고, 이를 검색하여 답변하는 전체 RAG 파이프라인을 구축합니다. 단계별로 코드를 나누어 진행합니다.

###

### 4.1 라이브러리 임포트 및 기본 설정

#### 4.1.1 라이브러리 임포트

```python
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

#### 4.1.2 환경 변수 로드 및 API 키 확인

```python
# 1. 환경 변수 로드 (.env 파일에서 API 키 가져오기)
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("오류: GEMINI_API_KEY를 찾을 수 없습니다. .env 파일을 확인해주세요.")
    exit()
```

#### 4.1.3 Gemini 클라이언트 및 로컬 LLM 초기화

```python
# 2. Google Gemini 클라이언트 초기화 (임베딩 생성용)
client = genai.Client(api_key=api_key)

# 3. 로컬 LLM 초기화 (답변 생성용)
# temperature=0: 사실 기반 답변을 위해 창의성을 낮춤
llm = ChatOllama(model="qwen3:0.6b", temperature=0)
```

***

### 4.2 핵심 함수 1: 텍스트 임베딩(Vectorization)

#### 4.2.1 임베딩 개념

* 텍스트 → 숫자 벡터로 변환
* 문서 저장용: `RETRIEVAL_DOCUMENT`
* 질문 검색용: `RETRIEVAL_QUERY`

#### 4.2.2 임베딩 함수 구현

```python
def get_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    """
    Google Gemini 모델을 사용하여 텍스트를 벡터로 변환합니다.
    - task_type: 'RETRIEVAL_DOCUMENT'(문서 저장용) 또는 'RETRIEVAL_QUERY'(검색 질문용)
    """
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"임베딩 오류 발생: {e}")
        return []
```

***

### 4.3 핵심 함수 2: 유사도 검색 (Retrieval)

#### 4.3.1 코사인 유사도 기반 검색

* 질문 벡터 vs 문서 벡터 간 내적(dot product) 사용
* 값이 클수록 의미적으로 더 유사

#### 4.3.2 검색 함수 구현

```python
def search_pdf(query, chunks, chunk_embeddings):
    """
    질문과 가장 유사한 문서 청크를 찾습니다.
    """
    # 1. 질문을 벡터로 변환 (검색 쿼리용 모드 사용)
    query_vec = get_embedding(query, task_type="RETRIEVAL_QUERY")
    
    if not query_vec:
        return "검색 오류"

    # 2. 코사인 유사도 계산 (질문 벡터 vs 모든 문서 벡터 내적)
    # 값이 클수록 유사도가 높음
    scores = [np.dot(query_vec, doc_vec) for doc_vec in chunk_embeddings]
    
    # 3. 가장 점수가 높은 인덱스 찾기
    best_idx = np.argmax(scores)
    
    return chunks[best_idx]
```

***

### 4.4 데이터 준비: PDF 로딩 및 벡터 DB 구축

#### 4.4.1 PDF 파일 로딩 및 텍스트 추출

```python
pdf_path = "src/2025_vla.pdf"  # 분석할 PDF 파일 경로

try:
    print(f"--- [데이터 준비 시작] ---")
    print(f"1. PDF '{pdf_path}' 로딩 중...")
    reader = PdfReader(pdf_path)
    
    # 텍스트가 있는 페이지만 추출하여 리스트로 저장
    chunks = [page.extract_text() for page in reader.pages if page.extract_text()]
    
    if not chunks:
        print("경고: PDF에서 텍스트를 추출할 수 없습니다. 이미지만 있는 PDF인지 확인하세요.")
        exit()
```

#### 4.4.2 페이지 단위 임베딩 생성

```python
    print(f"2. {len(chunks)}개 페이지 벡터화 진행 중... (시간이 조금 걸릴 수 있습니다)")
    # 모든 페이지를 순회하며 임베딩 생성
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    print("   -> 벡터화 완료! 검색 준비가 끝났습니다.\n")

except FileNotFoundError:
    print(f"오류: '{pdf_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit()
```

***

### 4.5 실전 테스트: RAG 성능 비교

ㄱ

1. 일반 LLM만 썼을 때
2. RAG를 적용했을 때

#### 4.6.1 테스트 질문 정의

```python
test_question = "이번 VLA 특강의 실습 조교(TA) 이름과 문의 이메일 주소를 알려줘."
```

***

#### 4.6.2 Case 1: RAG 미적용 (일반 LLM)

```python
print(f"--- [❌ Case 1: RAG 미적용 (일반 LLM)] ---")
print("모델이 학습 데이터에 없는 내용을 질문받았을 때:")

template_no_rag = "질문에 답변해 주세요.\n질문: {question}"
prompt_no_rag = ChatPromptTemplate.from_template(template_no_rag)
chain_no_rag = prompt_no_rag | llm | StrOutputParser()

# 모델이 할루시네이션(거짓말)을 하거나 모른다고 답하는지 확인
response_before = chain_no_rag.invoke({"question": test_question})
print(f"답변:\n{response_before}\n")
```

***

#### 4.6.3 Case 2: RAG 적용 (검색 + LLM)

**4.6.3.1 문서 검색 (Retrieval)**

```python
print(f"--- [⭕ Case 2: RAG 적용 (문서 검색 + LLM)] ---")
print("문서에서 근거를 찾아서 답변할 때:")

# 1. 검색 (Retrieval): 질문과 관련된 문서 내용 찾아오기
found_context = search_pdf(test_question, chunks, chunk_embeddings)
print(f"🔍 [검색된 근거 자료(Context)]:\n...{found_context[:150]}...\n")
```

**4.6.3.2 프롬프트 구성 (Augmentation)**

```python
rag_template = """
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
```

**4.6.3.3 최종 생성 (Generation)**

```python
# 3. 생성 (Generation): 정확한 답변 생성
rag_chain = rag_prompt | llm | StrOutputParser()
response_after = rag_chain.invoke({
    "context": found_context,   # 검색된 문서 내용
    "question": test_question   # 사용자 질문
})

print(f"최종 답변:\n{response_after}")
```
