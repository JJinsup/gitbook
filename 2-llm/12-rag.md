# \[12] RAG 실습

### 1.2 Ollama Python (RAG) 실습

Python 라이브러리를 활용하여 문서를 처리하고 Ollama 모델과 연동하는 RAG 워크플로우 예시입니다.

**사전 준비**

```
pip install langchain-ollama pypdf
```

#### 전체 순서 (Workflow)

1. **PDF 텍스트 추출**: PDF 파일에서 텍스트 데이터를 읽어옵니다. (Python 라이브러리 `pypdf` 사용)
2. **청킹 (Chunking)**: 긴 텍스트를 적당한 크기(예: 문단 단위)로 자릅니다. 이는 검색 정확도와 임베딩 효율을 높이기 위함입니다.
3. **문서 임베딩 (Indexing)**: 잘린 텍스트 조각들을 임베딩 모델(예: `text-embedding-004`)을 통해 벡터로 변환하여 벡터 저장소에 저장합니다.
4. **질문 임베딩 (Retrieval)**: 사용자의 질문(예: "이 문서 요약해줘")도 벡터로 변환하여, 저장된 문서 벡터 중 가장 유사도가 높은 내용을 찾습니다.
5. **답변 생성 (Generation)**: 찾아낸 관련 내용(Context)과 사용자의 질문을 합쳐 LLM(예: `gemini-2.5-flash-lite`)에게 보내 최종 답변을 생성합니다.
