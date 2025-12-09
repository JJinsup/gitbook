---
description: >-
  이 문서는 Google의 최신 google-genai 라이브러리를 사용하여 Python 환경에서 Gemini API를 활용하는 방법을
  다룹니다.
---

# \[9] GEMINI API 프롬프트 엔지니어링 실습

### 0. 대규모 언어 모델(LLM)이란?

API를 사용하기 전에, 우리가 다루는 대규모 언어 모델(Large Language Model)이 무엇인지 핵심만 간단히 짚고 넘어갑니다.

#### 0.1 정의 및 작동 원리

* **정의**: 방대한 양의 텍스트 데이터로 사전 학습된 **초대형 딥러닝 모델**입니다.
* **핵심 기술 (Transformer)**: 문장 속 단어들의 관계와 문맥을 이해하는 신경망 아키텍처입니다. 기존 방식(RNN)과 달리 데이터를 병렬로 처리하여 학습 속도와 성능이 비약적으로 높습니다.
* **단어 임베딩 (Word Embedding)**: 단어를 다차원 공간의 **숫자 벡터**로 변환하여, 의미가 비슷한 단어(예: 왕-남자, 여왕-여자)끼리 가깝게 배치해 문맥을 파악합니다.

#### 0.2 왜 중요한가요?

* **유연성**: 번역, 요약, 코딩, 창작 등 **하나의 모델**로 완전히 다른 여러 작업을 수행할 수 있습니다.
* **생성 능력**: 적은 양의 프롬프트(입력)만으로도 문맥에 맞는 자연스러운 결과물을 만들어내는 **생성형 AI**의 핵심입니다.

#### 0.3 학습 방식 3단계

1. **제로샷 (Zero-shot)**: 별도의 예시 없이 지시만으로 작업을 수행.
2. **퓨샷 (Few-shot)**: 몇 가지 예시를 제공하여 성능을 향상.
3. **미세 조정 (Fine-tuning)**: 특정 목적의 데이터를 추가로 학습시켜 모델을 최적화.

#### 0.4 주요 활용 분야

* **텍스트 생성**: 카피라이팅, 시나리오 작성, 이메일 초안 작성.
* **코드 생성**: 자연어를 코드로 변환 (예: "뱀 게임 만들어줘" -> Python 코드 생성).
* **요약 및 분류**: 긴 문서 요약, 고객 감정 분석, 정보 추출.

### 1. 시작하기 (Getting Started)

#### 1.1 라이브러리 설치

이 튜토리얼을 진행하기 위해서는 Google GenAI 라이브러리와 환경 변수 관리를 위한 `python-dotenv`가 필요합니다.

```
pip install google-genai python-dotenv
```

#### 1.2 API 키 발급 받기

1. [Google AI Studio](https://aistudio.google.com/)에 접속합니다.
2. Google 계정으로 로그인합니다.
3. **Get API key** 버튼을 클릭하여 새 API 키를 생성합니다.
4. 생성된 키(문자열)를 복사해둡니다. (절대 외부에 노출하지 마세요!)

#### 1.3 API 비율 제한 (Rate Limits)

Gemini API는 무료 등급과 유료 등급에 따라 사용량 제한이 다릅니다. 개발 시 이 제한을 염두에 두어야 합니다.

* **무료 등급 (Gemini 2.5 Flash 기준)**
  * **RPM (Requests Per Minute)**: 분당 15회 요청 가능
  * **RPD (Requests Per Day)**: 하루 1,500회 요청 가능
  * **TPM (Tokens Per Minute)**: 분당 100만 토큰 사용 가능
  * _주의: 무료 등급 데이터는 모델 개선을 위해 사용될 수 있습니다._
* **유료 등급 (Pay-as-you-go)**
  * **RPM**: 분당 2,000회 (이후 등급에 따라 증가)
  * **RPD**: 무제한 (비용 지불)
  * **데이터 프라이버시**: 입력 데이터가 모델 학습에 사용되지 않습니다.

> **⚠️ 429 에러 (Too Many Requests)**
>
> 코드를 실행하다가 `429` 에러가 발생한다면, 위에서 설명한 비율 제한(RPM/RPD)을 초과했다는 뜻입니다. 이 경우 잠시 기다렸다가 다시 실행하거나, `time.sleep()`을 사용하여 요청 간격을 조절해야 합니다.

### 2. 환경 변수 설정 (Environment Setup)

API 키를 코드에 직접 적는 것은 보안상 매우 위험합니다. `.env` 파일을 사용하여 안전하게 관리하는 방법을 알아봅니다.

#### 2.1 .env 파일 생성

프로젝트 최상위 폴더(Root Directory)에 `.env`라는 이름의 파일을 만들고 아래와 같이 작성합니다.

**파일: `.env`**

```
GEMINI_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### 2.2 .gitignore 설정 (필수!)

Git을 사용하여 버전을 관리한다면, `.env` 파일이 GitHub 등에 올라가지 않도록 반드시 `.gitignore` 파일에 추가해야 합니다.

**파일: `.gitignore`**

```
# 환경 변수 파일 무시
.env

# Python 컴파일 파일 무시
__pycache__/
*.pyc
```

#### 2.3 코드에서 API 키 불러오기

이제 Python 코드에서 안전하게 키를 불러와 클라이언트를 초기화합니다.

```
import os
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv()

# 환경 변수에서 키 가져오기
api_key = os.environ.get("GEMINI_API_KEY")

# 클라이언트 초기화
client = genai.Client(api_key=api_key)
```

### 3. Hello Gemini (기본 사용법)

가장 기본적인 텍스트 생성 요청을 보내봅니다.

```
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="오늘 저녁 메뉴 추천해줘",
    )
    print(response.text)
except Exception as e:
    print("키 설정이 잘못되었거나 에러가 발생했습니다:", e)
```

### 4. 시스템 지침 (System Instructions)

`System Instruction`은 AI에게 페르소나(인격)를 부여하거나 **행동 규칙**을 정해주는 강력한 기능입니다. 이를 통해 AI의 답변 스타일을 제어할 수 있습니다.

#### 4.1 튜토리얼 생성기 만들기

AI에게 "지침이 있으면 목록으로 만들고, 없으면 없다고 하라"는 규칙을 부여합니다.

```
sys_instruction = """
당신은 주어진 텍스트를 바탕으로 튜토리얼을 생성하는 AI 어시스턴트입니다.
텍스트에 어떤 절차를 진행하는 방법에 대한 지침이 포함되어 있다면, 
글머리 기호 목록 형식으로 튜토리얼을 생성하십시오.
그렇지 않으면 텍스트에 지침이 포함되어 있지 않음을 사용자에게 알리십시오.
"""

# 사용자 입력 (지침이 있는 경우)
text_input = """
먼저 평평한 땅을 골라 그라운드 시트를 깝니다. 그 위에 텐트 본체를 펼쳐주세요.
폴대를 조립해서 텐트의 슬리브(구멍)에 X자 모양으로 끼워 넣습니다.
폴대 끝을 텐트 모서리 아일렛(구멍)에 꽂아 텐트를 자립시킵니다.
마지막으로 팩을 45도 각도로 박아서 텐트를 바닥에 고정하고, 
플라이(덮개)를 씌워주면 완성입니다.
하지만 비가 너무 많이 오면 이 텐트는 사용할 수 없습니다.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", 
    config=genai.types.GenerateContentConfig(
        system_instruction=sys_instruction,
    ),
    contents=text_input,
)
print(response.text)
```

### 5. 실전 프롬프트 패턴

#### 5.1 단계별 생각 유도 (Chain of Thought)

AI에게 "생각하는 과정"을 출력하게 하면, 복잡한 요약이나 추론 문제에서 더 나은 결과를 얻을 수 있습니다.

```
sys_instruction = """
당신은 글을 요약하는 AI 어시스턴트입니다.
이 작업을 완료하려면 다음 하위 작업을 수행하십시오:

1. 제공된 글의 내용을 종합적으로 읽고 주요 주제와 핵심 요점을 식별합니다.
2. 현재 기사 내용을 요약하여 본질적인 정보와 주요 아이디어를 전달하는 단락 요약을 반드시 한글로 생성합니다.
3. 이 모든 사고 과정의 각 단계를 출력합니다.
"""

# (article 변수에는 긴 뉴스 기사나 텍스트가 들어갑니다)
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    config=genai.types.GenerateContentConfig(
        system_instruction=sys_instruction,
    ),
    contents=article,
)
```

#### 5.2 페르소나 부여 (아재개그 봇)

AI의 말투와 성격을 지정하여 재미있는 챗봇을 만들 수 있습니다.

```
sys_instruction = """
당신은 '아재개그'와 '넌센스 퀴즈'의 달인입니다.
사용자가 내는 문제의 숨겨진 말장난(Wordplay)을 파악하여 재치 있게 답변하세요.

[지침]
1. 너무 과학적이거나 논리적인 분석은 하지 마세요. (재미없습니다!)
2. 단어의 발음, 동음이의어, 글자 변형 등을 활용해 답을 찾으세요.
3. 정답을 먼저 외치고, 그 이유를 유머러스하게 덧붙이세요.
4. 말투는 약간 능글맞고 자신감 넘치게 하세요. (예: "정답은 바로~", "깔깔깔")
"""

riddle = "세상에서 가장 가난한 왕은?"

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    config=genai.types.GenerateContentConfig(
        system_instruction=sys_instruction,
    ),
    contents=riddle,
)
```

#### 5.3 전문가 모드 (클린 코드 & 자소서 컨설팅)

특정 분야의 전문가처럼 행동하도록 하여 고품질의 결과물을 얻습니다.

**예시 1: Python 클린 코드 생성기**

```
sys_instruction = """
당신은 파이썬 클린 코드(Clean Code) 전문가입니다.
사용자의 요청에 따라 PEP 8 스타일 가이드를 준수하는 고품질 파이썬 코드를 작성하세요.

[작성 규칙]
1. 모든 함수에는 Type Hint를 작성하세요.
2. 함수에는 기능을 설명하는 Docstring을 포함하세요.
3. 코드는 간결하고 효율적이어야 합니다.
"""
```

**예시 2: 자소서 업그레이드 (STAR 기법)**

```
sys_instruction = """
당신은 대기업 인사팀장 및 명문대 입학사정관 출신의 '자기소개서 전문 컨설턴트'입니다.
사용자가 자신의 경험이나 활동 내용을 거칠게 입력하면, 이를 평가자가 주목할 만한 '구체적인 역량'과 '성과' 중심의 문장으로 업그레이드해야 합니다.

[컨설팅 원칙]
1. STAR 기법 적용: 상황(S) -> 과제(T) -> 행동(A) -> 결과(R)
2. 구체적 수치화: '많이' 대신 '30% 증가' 등으로 표현
3. 전문 용어 사용
"""
```

### 📝 마치며

이 가이드에서는 Google Gemini API의 기본적인 사용법부터 시스템 지침을 활용한 다양한 프롬프트 엔지니어링 기법까지 알아보았습니다. `System Instruction`을 어떻게 작성하느냐에 따라 AI의 성능과 활용도가 무궁무진하게 달라집니다. 여러분만의 창의적인 지침을 만들어보세요!
