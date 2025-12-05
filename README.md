---
icon: hand-wave
layout:
  width: default
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: true
metaLinks:
  alternates:
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/
---

# 2025 동계 방학 VLA 로봇 특강

**"From Simulation to Reality: VLA 모델로 제어하는 로봇 팔"**

{% hint style="info" %}
#### Instructor & TA

* **주관:** 국민대학교 Wireless intelligent Technology LAB
* **책임 강사:** 주민철 교수님
* **실습 조교(TA):** 임진섭 (석사 과정), 김예찬 (학부 연구생)
* **문의:** `limsk519@kookmin.ac.kr`
* **최종 수정일 : 2025.12.04**
{% endhint %}

본 GitBook은 2025년 동계 방학 동안 진행되는 학부생 대상 로봇 제어 특강의 강의 자료 및 가이드라인을 담고 있습니다.&#x20;

### 📅 Overview

* **기간:** 2025.12.22 (월) \~ 2026.01.13 (화)
* **대상:** 학부생 (4개 조 운영)
* **주요 목표:**
  1. **VLA 이해:** Vision-Language-Action 모델의 최신 트렌드 습득
  2. **Simulation:** MuJoCo & NVIDIA Isaac Sim 환경 구축 및 실습
  3. **Sim-to-Real:** 가상 환경 데이터로 실물 로봇(SO-ARM 101) 제어

### 🗓️ 세부 일정 및 커리큘럼

#### \[1주 차] 연구 개발 환경 구축 (On-Boarding)

**"AI/로봇 연구를 위한 첫걸음, 리눅스 환경 세팅"**

* **일정:** 12.22(월) \~ 12.26(금)
* **학습 목표:** 리눅스(Ubuntu 22.04) 기반의 개발 환경 완비

| 구분                | 내용                                     | 비고               |
| ----------------- | -------------------------------------- | ---------------- |
| **OS 설치**         | 개인 노트북 사양에 따른 맞춤형 설치                   | Ubuntu 22.04 LTS |
| **Option A**      | **WSL2 (Windows Subsystem for Linux)** | 접근성 및 안정성 중시     |
| **Option B (추천)** | **Native Dual Booting**                | 로컬 하드웨어 성능       |
| **환경 설정**         | 실습 환경 세팅                               |                  |

#### \[2주 차] VLA 모델 기초 및 MuJoCo 시뮬레이션

**"가벼운 모델로 시작하는 로봇 제어와 물리 엔진"**

* **일정:** 12.29(월) \~ 12.30(화)
* **학습 목표:** 경량화 VLA 모델(SmolVLA) 구동 및 MuJoCo 물리 엔진 이해

**12.29 (월): 물리 엔진 기초**

* **MuJoCo 기초:** 로봇 시뮬레이션의 기본이 되는 물리 엔진 실습
* **LLM API 활용:** 자연어 명령을 통한 로봇 제어 맛보기 (Prompt Engineering)

**12.30 (화): On-Device VLA 실습**

* **모델:** 최신 경량 모델 **\[SmolVLA (0.45B)]** 활용
* **실습:** 개인 PC/노트북 로컬 GPU를 활용한 Inference(추론) 및 Fine-tuning 기초

#### \[3주 차] 강화학습(RL)과 NVIDIA Isaac Sim

**"시뮬레이션과 강화학습의 만남"**

* **일정:** 01.05(월) \~ 01.06(화)
* **학습 목표:** Isaac Lab을 활용한 고사양 시뮬레이션 및 강화학습 적용

**01.05 (월): 강화학습(RL) 이론**

* **핵심 이론:** MDP (Markov Decision Process), DQN 등
* **설계:** 로봇 제어를 위한 보상 함수(Reward Function) 설계 방법

**01.06 (화): Isaac Lab 조별 실습 (Server-Client)**

* **서버(Server):** 조별로할당받은 GPU에서 Isaac Sim 구동 \
  (Headless/Streaming 모드)
* **클라이언트(Client):** 학생 개인 PC에서 **Omniverse Client**로 접속하여 원격 실습

#### \[4주 차] Sim-to-Real 프로젝트

**"가상을 넘어 현실로: 하드웨어 제어 및 배포"**

* **일정:** 01.12(월) \~ 01.13(화)
* **학습 목표:** 시뮬레이션 데이터 수집부터 실물 로봇 제어까지의 전체 파이프라인 완성

**01.12 (월): 하드웨어 및 데이터 파이프라인**

* **하드웨어:** 오픈소스 로봇 팔 **SO-ARM 101** 조립 및 구동 원리 파악
* **데이터 수집:** **\[LeIsaac]** 프레임워크 활용
  * Isaac Sim 가상 환경에서 실물용 데이터셋 자동 수집 파이프라인 구축

**01.13 (화): VLA 파인튜닝 및 최종 시연**

* **모델 학습:** 수집된 데이터(Sim & Real)를 활용하여 **SmolVLA** 모델 \
  LoRA(Low-Rank Adaptation) 학습
* **Sim-to-Real:** 학습된 모델을 로컬 PC에 배포하여 실물 로봇 제어 미션 수행 및 시연

### 🛠️ 준비물 및 선수 지식

* **준비물:** 개인 노트북 (NVIDIA GPU 권장, 없을 시 서버 API로 대체)
* **선수 지식:** Python 기초, Linux 기본 명령어, 딥러닝 기초 이해

> 💡 **Note for Students** 본 특강은 실습 위주로 진행됩니다. 1주 차 환경 설정이 완료되지 않으면 이후 실습 진행이 어려우므로, **1주 차 가이드**를 꼼꼼히 따라와 주세요.
