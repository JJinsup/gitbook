---
description: >-
  이 장의 목표는 데이터 수집 → 서버 학습 → 로컬 추론이라는 전체 파이프라인을 직접 끝까지 완주, 이 실습을 통해 “정책 학습(Policy
  Learning)이 실제로 무엇을 의미하는지” 체감하는 것이 핵심입니다.
icon: brain-circuit
---

# \[23] MuJoCo: Train & Deploy ACT

{% embed url="https://huggingface.co/docs/lerobot/act" %}

### 23.1 실습 목표와 범위

| 구분      | 이번 장에서 수행하는 것               | 이번 장에서 하지 않는 것         |
| ------- | --------------------------- | ---------------------- |
| **환경**  | MuJoCo 시뮬레이터 내 로봇 조작        | 고성능 모델 하이퍼파라미터 튜닝      |
| **데이터** | Offline 데이터 수집 (성공 에피소드 저장) | 실시간 언어(Language) 입력 사용 |
| **학습**  | 서버 GPU를 이용한 ACT 정책 학습       | -                      |
| **배포**  | 학습된 모델을 로컬 환경에서 추론          | -                      |

{% hint style="info" %}
**중요**: 우리의 목표는 모델의 성능을 극대화하는 것이 아니라, **파이프라인의 전체 흐름을 완주**하는 것입니다.
{% endhint %}

### 23.2 데이터 수집 (Local)

#### 23.2.1 실행 방식 이해

데이터 수집 스크립트는 `NUM_DEMO` 값만큼 에피소드를 자동으로 연속 수집합니다. 즉, 한 번 실행하면 설정된 횟수만큼 성공할 때까지 데이터가 누적 저장됩니다.

```
# 데이터 수집 스크립트 실행
python 1.collect_data_standalone.py
```

<figure><img src="../.gitbook/assets/image (42).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/image (43).png" alt=""><figcaption></figcaption></figure>

<p align="center"><strong>생성된 데이터</strong></p>

#### 23.2.2 코드 설명 (`1.collect_data_standalone.py`)

**1) 주요 설정 (Configuration)**

학생들이 가장 먼저 확인하고 수정해야 할 부분입니다.

* `TASK_NAME`: 데이터셋에 붙는 라벨 (예: 'Put mug cup on the plate')
* `NUM_DEMO`: **“몇 판 성공할 때까지 모을 거냐”** (최소 30\~50 권장)
* `ROOT`: 데이터가 물리적으로 저장될 경로 (`./demo_data`)
* `xml_path`: 어떤 시각적 환경(MuJoCo Scene)에서 실습할지 지정

**2) 데이터 관리 로직**

* `ROOT` 폴더가 이미 존재하면 삭제할지(`y`), 기존 데이터에 이어붙일지(`n`)를 묻습니다.
* **처음 실습**: `y`로 지우고 깨끗하게 시작하세요.
* **데이터 보강**: `n`으로 설정하여 에피소드를 추가 수집하세요.

**3) 수집 메커니즘 (Mapping)**

학생들에게는 아래와 같이 데이터 구조를 매핑하여 설명하면 이해가 빠릅니다.

* **Observation**: 로봇이 본 것(카메라 이미지)과 현재 상태(관절 위치)
* **Action**: 사람이 조작한 행동(키보드 입력에 따른 관절 움직임)
* **Task**: 이 데이터가 도달하고자 하는 목표 문장

{% hint style="info" %}
**데이터 수집 팁**

* **Z 키**: 실수했을 때 환경을 초기화(Reset)합니다.
* **Record Flag**: 로봇이 가만히 있을 때는 저장하지 않다가, 실제 움직임이 감지되면 자동으로 기록을 시작합니다.
{% endhint %}

### 23.3 데이터 시각화 (Local)

수집한 데이터가 모델의 "재료"로서 적합한지 직접 확인합니다.

```
python 2.visualize_data_standalone.py
```

* **확인 사항**: 로봇의 움직임이 부드러운가? 카메라 이미지(Agent/Wrist)가 잘 찍혔는가?
* **관찰 포인트**: 타임스텝별로 변화하는 `state`와 `action`의 관계를 확인하며 “데이터가 정책의 재료”라는 감각을 익히세요.

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 21시 56분 21초.gif" alt=""><figcaption></figcaption></figure>

### 23.4 모델 학습 (Server)

데이터가 준비되었다면, 이제 GPU 자원을 활용해 정책을 학습시킬 차례입니다.

> _**\[22]를 참고해서 서버에도 대표 학생 한명이 환경 세팅을 해주세요**_

#### 23.4.1 데이터 업로드

로컬에서 생성된 `demo_data/` 폴더를 서버의 `lerobot-mujoco-tutorial` 프로젝트 경로 안으로 업로드

#### 23.4.2 학습 실행

서버에서 아래 노트북 파일을 실행합니다.

* **파일명**: `3.train.ipynb`
* **내용**: ACT(Action Chunking Transformer) 정책 학습
* **확인**: 학습 완료 후 `ckpt/` 폴더 내에 모델 가중치 파일이 생성되었는지 확인합니다.

#### 23.4.3 코드 이해



### 23.5 모델 배포 및 추론 (Local)

서버에서 학습된 `ckpt` 파일을 로컬로 다운로드한 후, 로봇이 스스로 움직이는지 확인합니다.

```
# 추론 스크립트 실행
python 4.deploy_standalone.py
```

<figure><img src="../.gitbook/assets/Screencast from 2026년 01월 14일 22시 03분 25초.gif" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
**반드시 짚고 넘어가야 할 포인트**

이 추론 결과는 고도의 '지능'이 아닙니다. 이것은 사람이 만든 데이터를 흉내 낸 결과일 뿐입니다.
{% endhint %}

* 새로운 상황(물체 위치 변경 등)에는 매우 취약합니다.
* 조건이 조금만 바뀌어도 쉽게 실패합니다.
* **이 한계를 이해하는 것**이 다음 장인 \[24] smolVLA 실습(언어 조건부 VLA)으로 넘어가는 이유입니다.

### 🏁 정리

> **ACT 실습은 “정책 학습의 전체 파이프라인을 직접 완주해보는 기초 연습”입니다.**
