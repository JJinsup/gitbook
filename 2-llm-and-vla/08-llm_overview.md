---
description: From "Attention Is All You Need" to GPT
---

# \[8] LLM Overview

### 1. 패러다임의 전환 (Paradigm Shift)

#### 1.1 기존의 한계: RNN의 구조적 문제점

트랜스포머 등장 이전, 자연어 처리(NLP) 분야는 인간의 언어와 같이 순서가 있는 시퀀스 데이터를 처리하기 위해 고안된 **RNN(Recurrent Neural Network, 순환 신경망)** 계열의 모델이 주도하고 있었습니다. RNN은 이전 시점(Time Step)의 정보를 현재 시점의 입력과 함께 처리하여 '기억(Memory)'을 유지한다는 혁신적인 개념을 도입했으나, 그 구조적 특성으로 인해 다음과 같은 치명적인 한계를 지니고 있었습니다.

* **순차적 처리(Sequential Processing)의 비효율**: RNN은 _**t**_ 시점의 은닉 상태($$h_t$$)를 계산하기 위해 반드시 이전 시점의 은닉 상태($$h_{t-1}$$)가 필요합니다.
  * 즉, 문장의 첫 번째 단어부터 마지막 단어까지 **순서대로 하나씩** 연산해야 합니다.
  * 이는 현대 딥러닝의 핵심인 **GPU의 병렬 연산(Parallelization)** 능력을 전혀 활용하지 못하게 만들어, 데이터의 길이가 길어질수록 학습 속도가 현저히 저하되는 원인이 되었습니다.
* **장기 의존성(Long-term Dependency) 및 기울기 소실(Vanishing Gradient)**:
  * RNN은 역전파(Backpropagation) 과정에서 시간을 거슬러 올라가며 가중치를 업데이트하는 **BPTT(Backpropagation Through Time)** 방식을 사용합니다.
  * 시퀀스가 길어질수록, 역전파되는 기울기(Gradient)가 계속해서 곱해지며 0에 수렴해버리는 '기울기 소실' 문제가 발생합니다.
  * 이로 인해 모델은 먼 과거의 정보(초반부 입력)를 잊어버리게 되며, 긴 문맥을 파악해야 하는 번역이나 요약 태스크에서 성능이 급격히 저하되는 한계를 보였습니다. $$f(x) = x * e^{2 pi i \xi x}$$

#### 1.2 트랜스포머의 해결책: "Attention Is All You Need"

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-09 17-05-40.png" alt=""><figcaption></figcaption></figure>

2017년 발표된 트랜스포머 논문은 이러한 순차적 처리 방식의 한계를 극복하기 위해, **'Recurrence(순환)'를 완전히 배제한** 새로운 아키텍처를 제안했습니다.

* **Recurrence 제거 및 완전 병렬화(Parallelization)**: 트랜스포머는 입력된 문장 전체를 행렬(Matrix)로 변환하여 **한 번에** 처리합니다. 순서대로 기다릴 필요가 없으므로 GPU 자원을 100% 활용하여 학습 속도를 비약적으로 향상시켰습니다.
* **Self-Attention을 통한** O(1) **거리 확보**: 문장 내 모든 토큰이 서로를 **동시에** 참조(All-to-All Connection)합니다. 문장의 맨 앞 단어와 맨 뒤 단어 사이의 거리가 물리적으로 멀더라도, 트랜스포머에서는 단 한 번의 연산으로 직접 연결됩니다. 이를 통해 장기 의존성 문제를 근본적으로 해결했습니다.

### 2. 입력의 준비: 전처리 과정 (Preprocessing)

모델이 자연어를 처리하기 위해서는 텍스트를 수치 데이터로 변환하는 과정이 선행되어야 합니다.

#### 2.1 토크나이저 (Tokenizer)

입력된 문장을 모델이 처리할 수 있는 최소 단위인 '토큰(Token)'으로 분절하고, 각 토큰에 고유한 ID를 부여하는 과정입니다.

* **Character-level (문자 단위)**: 구현이 간단하지만, 시퀀스 길이가 과도하게 길어지고 의미 파악이 어렵다는 단점이 있습니다.
* **Subword-level (부분 단어 단위)**: BPE(Byte Pair Encoding) 등의 알고리즘을 사용하여, 빈도수가 높은 단어는 하나로 묶고 희귀한 단어는 분절합니다. 현대 LLM은 주로 이 방식을 채택하여 효율성과 의미 보존의 균형을 맞춥니다.

#### 2.2 임베딩 (Embedding)

각 토큰의 ID를 고차원 벡터 공간의 좌표로 변환하여 의미(Semantics)를 부여하는 과정입니다.

* **벡터화**: 단순한 정수 인덱스를 연속적인 벡터 값으로 매핑합니다.
* **의미 공간**: 잘 학습된 임베딩 공간에서는 의미적으로 유사한 단어(예: '사과'-'배')가 가까운 거리에 위치하게 됩니다. 이를 통해 모델은 단어 간의 의미적 관계를 수학적으로 연산할 수 있게 됩니다.

### 3. 핵심 메커니즘: Self-Attention

트랜스포머의 가장 핵심적인 구성 요소로, 입력 시퀀스 내의 각 토큰이 다른 모든 토큰과의 관계(유사도)를 계산하여 문맥 정보를 추출하는 메커니즘입니다.

#### 3.1 Q, K, V의 개념

데이터베이스 검색 시스템의 개념을 차용하여, 각 토큰을 세 가지 벡터로 투영(Projection)합니다.

* **Query (Q)**: 현재 시점에서 필요한 정보를 요청하는 벡터입니다. ("무엇을 찾고자 하는가?")
* **Key (K)**: 각 토큰이 가지고 있는 고유한 식별 정보를 담은 벡터입니다. ("나는 어떤 정보를 가지고 있는가?")
* **Value (V)**: 각 토큰의 실제 콘텐츠 정보를 담은 벡터입니다. ("나의 실제 값은 무엇인가?")

#### 3.2 Scaled Dot-Product Attention

Self-Attention의 수식은 다음과 같이 정의됩니다.

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

1. $$QK^T$$ **(유사도 계산)**: Query와 Key의 내적(Dot Product)을 통해 두 벡터 간의 유사도를 계산합니다.
2. $$\frac{1}{\sqrt{d_k}}$$**(Scaling)**: 내적 값이 커질수록 Softmax 함수에서 기울기가 0에 가까워지는 현상(Vanishing Gradient)을 방지하기 위해, 차원의 제곱근으로 나누어 스케일링을 수행합니다.
3. **Softmax (정규화)**: 계산된 유사도 점수를 0과 1 사이의 확률값으로 변환하여, 각 토큰에 얼마나 집중(Attention)할지 결정합니다.
4. _**V**_ **가중합**: 계산된 Attention 가중치를 Value 벡터에 곱하여, 문맥이 반영된 최종 벡터를 생성합니다.

### 4. 모델 구조의 심화 (Architecture Details)

단순한 Attention 메커니즘을 넘어, 깊은 신경망 학습을 가능하게 하는 구조적 장치들입니다.

#### 4.1 Multi-Head Attention

* **개념**: $$d_{model}$$ 차원의 벡터를 _**h**_&#xAC1C;의 헤드(Head)로 분할하여 병렬로 Attention을 수행하는 방식입니다.
* **효과**: 단일 Attention으로는 포착하기 어려운 다양한 관점(예: 문법적 관계, 의미적 관계, 위치적 관계 등)의 정보를 동시에 학습할 수 있게 하여 모델의 표현력을 풍부하게 만듭니다.

#### 4.2 Position-wise Feed-Forward Networks (FFN)

* **역할**: Multi-Head Attention을 통해 수집된 문맥 정보를 각 토큰별로 개별적으로 가공하는 비선형 변환 층입니다.
* **구조**: 두 개의 선형 변환(Linear Transformation)과 그 사이의 활성화 함수(주로 ReLU 또는 GELU)로 구성됩니다.

#### 4.3 Residual Connection & Layer Normalization

* **Residual Connection (잔차 연결)**: $$Output = F(x) + x$$의 형태로, 하위 층의 정보를 상위 층으로 직접 전달합니다. 이는 깊은 네트워크에서도 정보 소실 없이 학습이 가능하게 하는 핵심 요소입니다.
* **Layer Normalization**: 각 층의 출력을 정규화하여 학습 과정을 안정화하고 수렴 속도를 가속화합니다.

### 5. 인코더와 디코더 (Encoder & Decoder)

논문의 트랜스포머는 기계 번역을 위해 인코더-디코더 구조를 채택했습니다.

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

#### 5.1 인코더 (Encoder)

* 입력 시퀀스의 정보를 압축하여 문맥 벡터로 변환하는 역할을 합니다.
* **Bidirectional Attention**: 문장의 앞뒤 문맥을 모두 참조할 수 있어, 문장 이해(NLU) 태스크(예: BERT)에 적합합니다.

#### 5.2 디코더 (Decoder)

* 인코더의 정보를 바탕으로 타겟 시퀀스를 생성하는 역할을 합니다.
* **Masked Self-Attention**: 생성 모델의 특성상(Auto-regressive), 현재 시점보다 미래의 정보를 참조해서는 안 됩니다. 따라서 미래 시점의 토큰에 대한 Attention Score를 $$\infty$$로 마스킹하여 정보 접근을 차단합니다. 이는 GPT 계열 모델의 핵심 메커니즘입니다.

### 6. GPT로의 진화 (Evolution to GPT)

GPT는 초기 버전(GPT-1\~3)에서 Transformer의 디코더 블록만을 사용한 언어모델로 시작했지만, 최신 GPT 모델들은 순수 디코더 구조를 넘어서는 확장된 아키텍처를 사용합니다.

#### 6.1 Decoder-only Architecture

* **목적**: GPT는 번역이 아닌 "다음 토큰 예측(Next Token Prediction)"을 목표로 합니다.
* **원리**: 대규모 텍스트 데이터(Corpus)를 비지도 학습(Unsupervised Learning)하여, 주어진 문맥 뒤에 올 가장 확률 높은 단어를 예측하는 능력을 갖추게 됩니다.

#### 6.2 모델의 발전 양상

* 초기 모델(n-gram 등)은 문맥 파악 능력이 현저히 떨어졌으나, 트랜스포머 기반의 GPT는 Multi-Head Attention과 깊은 레이어를 통해 긴 문맥과 복잡한 언어적 뉘앙스를 완벽하게 학습할 수 있게 되었습니다.

### 7. 인간 피드백 기반 강화학습 (RLHF)

{% embed url="https://openai.com/ko-KR/index/instruction-following/#guide" %}

GPT-3와 같은 초기 LLM은 유창한 텍스트 생성 능력은 갖추었으나, 사용자의 의도에 부합하거나 안전한 답변을 생성하는 데에는 한계가 있었습니다. 이를 보완하기 위해 RLHF(Reinforcement Learning from Human Feedback)가 도입되었습니다.

RLHF는 크게 3단계 프로세스로 진행됩니다.

#### 7.1 Step 1: Supervised Fine-Tuning (SFT)

<p align="center"><img src="../.gitbook/assets/image (10).png" alt=""></p>

* **지도 미세 조정**: 인간 라벨러가 작성한 양질의 '질문-답변' 쌍 데이터를 모델에 학습시킵니다. 이를 통해 모델은 인간이 선호하는 답변의 형식과 톤을 모방하게 됩니다.

#### 7.2 Step 2: Reward Model Training (RM)

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

* **보상 모델 학습**: 모델이 생성한 여러 답변 후보에 대해 인간이 선호도 순위(Ranking)를 매깁니다. 이 데이터를 바탕으로, 어떤 답변이 더 우수한지 점수를 예측하는 별도의 '보상 모델'을 학습시킵니다.

#### 7.3 Step 3: Proximal Policy Optimization (PPO)

<figure><img src="../.gitbook/assets/image (12).png" alt=""><figcaption></figcaption></figure>

* **강화학습 적용**: 생성 모델(Policy)이 답변을 생성하면 보상 모델이 점수(Reward)를 부여하고, 이 점수를 최대화하는 방향으로 생성 모델을 업데이트합니다. PPO 알고리즘은 학습 과정에서 정책이 급격하게 변하는 것을 방지하여 안정적인 최적화를 돕습니다.

### 8. Conclusion

트랜스포머 아키텍처는 **병렬 처리**와 **Attention 메커니즘**을 통해 NLP 분야의 기술적 난제를 해결하였으며, 이는 현대 AI 모델의 표준이 되었습니다. 나아가 **RLHF**와 같은 정렬(Alignment) 기술의 도입으로, 단순한 언어 모델을 넘어 인간의 의도를 이해하고 상호작용하는 인공지능으로 진화하고 있습니다.
