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

# 2026 동계 방학 VLA 특강

**"From Simulation to Reality: VLA 모델로 제어하는 로봇 팔"**

{% hint style="info" %}
#### Instructor & TA

* **주관:** 국민대학교 Wireless intelligent Technology LAB
* **책임 강사:** 주민철 교수님
* **실습 조교(TA):** 임진섭 (석사 과정), 김예찬 (학부 연구생)
* **문의:** `limsk519@kookmin.ac.kr`
* **최종 수정일 : 2025.12.18**
{% endhint %}

본 GitBook은 2025-26 동계 방학 동안 진행되는 학부생 대상 로봇 제어 특강의 강의 자료 및 가이드라인을 담고 있습니다.&#x20;

### 📅 Overview

* **기간:** 2025.12.22 (월) \~ 2026.01.13 (화)
* **대상:** 학부생 (4개 조 운영)
* **주요 목표:**
  1. **VLA 이해:** Vision-Language-Action 모델의 최신 트렌드 습득
  2. **Simulation:** MuJoCo & NVIDIA Isaac Sim 환경 구축 및 실습
  3. **Sim-to-Real:** 가상 환경 데이터로 실물 로봇(SO-ARM 101) 제어

## 🗓️ 세부 일정 및 커리큘럼

#### \[Level 1] 연구 개발 환경 구축 (On-Boarding)

**"AI/로봇 연구를 위한 첫걸음, 리눅스 환경 세팅"**

* **일정:** \~ 12.22(월)
* **학습 목표:** 리눅스(Ubuntu 22.04) 기반의 개발 환경 완비

| 구분                | 내용                                     | 비고               |
| ----------------- | -------------------------------------- | ---------------- |
| **OS 설치**         | 개인 노트북 사양에 따른 맞춤형 설치                   | Ubuntu 22.04 LTS |
| **Option A**      | **WSL2 (Windows Subsystem for Linux)** | 접근성 및 안정성 중시     |
| **Option B (추천)** | **Native Dual Booting**                | 로컬 하드웨어 성능       |

### \[WEEK 1] MuJoCo & LLM 기초 (12.22 \~ 12.23)

**"물리 엔진과 거대언어모델(LLM)의 만남"**

* **일정:** 12.22(월) \~ 12.23(화)
* **학습 목표:** 로봇 시뮬레이션의 기초를 다지고, LLM을 활용한 로봇 제어 가능성 탐구

**\[LEVEL 2] 오리엔테이션 및 MuJoCo 시뮬레이션 (12.22 월)**

* **Intro:** 전체 특강 계획 설명 및 팀 구성
* **MuJoCo 기초:** \* MuJoCo 물리 엔진의 원리 이해 (MJCF 모델링)
  * Viewer 조작 및 기본 시뮬레이션 실습

**\[LEVEL 3] LLM 실습 (12.23 화)**

* **LLM 이해:** OpenAI / Gemini API 활용법 및 Prompt Engineering 기초
* **LLM for Robotics:** 자연어 명령을 로봇 제어 코드(Python)로 변환하는 실습

#### \[WEEK 2] VLA + MuJoCo 집중 탐구 (12.29 \~ 12.30)

**"Vision-Language-Action 모델 실전 적용"**

* **일정:** 12.29(월) \~ 12.30(화)
* **학습 목표:** 주행 로봇과 로봇 팔 각각에 적합한 VLA 모델을 시뮬레이션과 연동

**\[LEVEL 4] VLA 실습 I : 주행 로봇 (12.29 월)**

* **Target:** **Mobile Robot (TurtleBot3)**
* **Tech Stack:** **Gemini (Vision-Language) + YOLO (Object Detection)**
* **Mission:** MuJoCo 환경 내에서 주행 로봇이 자연어 명령을 수행하도록 제어

**\[LEVEL 4] VLA 실습 II : 로봇 팔 (12.30 화)**

* **Target:** **Robot Arm (Manipulator)**
* **Tech Stack:** **SmolVLA (OpenVLA 경량화 모델)**
* **Mission:** SmolVLA 모델을 MuJoCo와 연동하여 로봇 팔 제어 및 Fine-tuning 기초 실습

#### \[WEEK 3] NVIDIA Isaac Sim & Lab (01.05 \~ 01.06)

**"고해상도 시뮬레이션과 강화학습 환경 구축"**

* **일정:** 01.05(월) \~ 01.06(화)
* **학습 목표:** NVIDIA 생태계(Omniverse)를 활용한 고도화된 시뮬레이션 환경 구축

**\[LEVEL 5] Isaac Sim 기초 및 서버 활용 (01.05 월)**

* **Isaac Sim 입문:** USD 포맷 이해 및 환경(Stage) 구성
* **Server Setting:** 연구실 **RTX 3090 서버**에서 Headless/Streaming 모드로 Isaac Sim 구동하는 법

**\[LEVEL 6] Isaac Lab 심화 (01.06 화)**

* **Isaac Lab:** Isaac Sim 기반의 로봇 학습 프레임워크 실습
* **RL Environment:** 강화학습을 위한 로봇 환경 설정 및 학습 예제 실행

#### \[WEEK 4 \~] Sim-to-Real 프로젝트 (01.12 \~)

**"가상을 넘어 현실로: 하드웨어 제어 및 배포"**

* **일정:** 01.12(월) \~ 01.13(화)
* **학습 목표:** 시뮬레이션 데이터 수집부터 실물 로봇 제어까지의 전체 파이프라인 완성

**\[LEVEL 7] 팀 프로젝트 Kick-off: SO-ARM 101 (01.12 월 \~)**

* **Team Project:** 4주 차부터는 팀 단위 프로젝트로 진행되며, **LeIsaac**과 **SmolVLA**를 활용한 Sim-to-Real 구현이 최종 목표
* **Hardware:** 오픈소스 로봇 팔 **SO-ARM 101** 조립 및 구동
* **Framework 1:** [**LeIsaac**](https://github.com/LightwheelAI/leisaac) (LightwheelAI)
  * Teleoperation 환경 구축 및 VR/컨트롤러 기반 데이터 수집
* **Framework 2 (Final Goal):** [**SmolVLA**](https://huggingface.co/docs/lerobot/smolvla) (HuggingFace LeRobot)
  * 수집된 데이터로 SmolVLA 모델 학습 및 실물 로봇 제어(Sim-to-Real) 달성

### :computer: 문서 작성 하드웨어 스펙

_**로컬 PC ( Ubuntu 22.04)**_

* CPU : 13th Gen Intel(R) Core(TM) i7-13700
* GPU : NVIDIA GeForce GTX 1650 4GB

_**서버 ( Ubuntu 22.04)**_

* GPU : 3090 24GB 1개만 사용

#### _**시뮬레이션 실습 코드**_&#x20;

{% embed url="https://github.com/JJinsup/mujoco_llm" %}

{% embed url="https://github.com/JJinsup/lerobot-mujoco-tutorial" %}

{% embed url="https://github.com/JJinsup/so_arm_101_isaac" %}

#### 참고 사이트

{% embed url="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/quickstart_index.html" %}

{% embed url="https://isaac-sim.github.io/IsaacLab/main/index.html" %}

{% embed url="https://lycheeai-hub.com/" %}

{% embed url="https://wiki.seeedstudio.com/training_soarm101_policy_with_isaacLab/" %}

{% embed url="https://wikidocs.net/book/18629" %}

### 🛠️ 준비물 및 선수 지식

* **준비물:** 개인 노트북 (NVIDIA GPU 권장, 없을 시 코랩으로 대체)
* **선수 지식:** Python 기초, Linux 기본 명령어, 딥러닝 기초 이해

> 💡 **Note for Students** 본 특강은 실습 위주로 진행됩니다. **\[사전 준비]** 단계인 리눅스 환경 설정이 완료되지 않으면 수업 참여가 어려우므로 첫 수업 전까지 반드시 완료해 주세요.
