---
description: >-
  본 서버는 smolVLA 모델 학습과 Jetson 배포를 목적으로 운영됩니다. 4개 조가 2장의 RTX 3090(24GB)을 효율적으로
  공유하기 위해 아래 규칙을 반드시 준수해야 합니다.
icon: landmark-magnifying-glass
---

# \[A] 서버 사용법

### 0. 서버 접속 방법

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

1. VSCODE에서 `Remote Explorer > Open SSH Config File`을 클릭한다.
2. /home/$USERNAME/.ssh/config파일을 클릭한다.

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

3. 아래 예시처럼 입력한다. **(별명, IP주소, 유저 이름, 포트를 자신의 환경에 맞게 변경한다)**

```
Host [별명]
    HostName [서버 IP 주소]
    User [유저이름]
    Port [접속할 포트]
    RequestTTY yes
```

### 1. 저장소 관리 규칙

우리 서버는 OS용 SSD와 데이터용 대용량 SSD(`/data2`)가 물리적으로 분리되어 있습니다.

| 구분       | 경로 (Directory)    | 용량       | 주 목적                   | 주의사항                    |
| -------- | ----------------- | -------- | ---------------------- | ----------------------- |
| **OS**   | `/`               | 2TB      | 코드(`.py`), 설정 파일, 가상환경 | 대용량 파일 저장 시 서버 다운 유발    |
| **Data** | `/data2/studentX` | **15TB** | 데이터셋, 체크포인트, 가중치       | **모든 큰 파일은 반드시 이곳에 저장** |

{% hint style="info" %}
**핵심 요약**

**/home**: 텍스트 파일(코드) 위주로 관리

**/data2/student**: 무거운 파일(이미지, 모델 weight) 저장
{% endhint %}

### 2. 파이썬 및 패키지 환경 (Conda)

시스템 전체 환경을 오염시키지 않기 위해 모든 작업은 **Conda 가상환경**에서 수행합니다.

#### 가상환경 생성 및 활성화

```
# 가상환경 생성
conda create -n smolvla python=3.10

# 가상환경 활성화
conda activate smolvla
```

{% hint style="info" %}
**sudo pip 사용 절대 금지** \
`sudo pip install`은 시스템 파이썬 경로를 수정하여 모든 사용자의 환경을 망가뜨립니다.
{% endhint %}

1. 반드시 가상환경 활성화 후 `pip install [패키지명]`(sudo 없이)만 사용하세요.
2. 설치 전 `which pip`를 입력해 내 계정의 conda 경로가 맞는지 확인하세요.&#x20;

### 3. GPU 자원 활용

모든 조는 GPU 0번과 1번에 접근할 수 있습니다. 유연하게 사용하되 충돌을 방지해야 합니다.

#### A. 실행 시 GPU 지정 (권장)

코드 수정 없이 실행 시 환경 변수로 사용할 GPU를 지정하세요.

**\[Terminal] 터미널에서 실행할 때**

코드 수정 없이 환경 변수로 GPU를 지정합니다.

* **0번 GPU 사용**: `CUDA_VISIBLE_DEVICES=0 python train.py`
* **1번 GPU 사용**: `CUDA_VISIBLE_DEVICES=1 python train.py`
* **Dual GPU 사용**: `CUDA_VISIBLE_DEVICES=0,1 python train.py`&#x20;

**\[Jupyter] .ipynb 파일에서 실행할 때**

코드의 최상단(import torch 이전)에 아래 내용을 추가하세요.

```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 번호 입력
```

#### B. 선(先) 확인 후(後) 실행

학습을 시작하기 전, 반드시 현재 사용 중인 GPU가 있는지 확인하세요.

* **확인 방법**: `nvtop` 또는 `nvidia-smi` 실행
* **판단 기준**: VRAM이 1GB 이상 사용 중이라면 다른 조가 작업 중인 것입니다.

### 4. 운영 규칙 및 에티켓

* 📅 **시간 예약제**: 코드 디버깅은 상시 가능하나, **30분 이상의 파인튜닝 작업**은 조별 공유된 \[예약 캘린더]에 슬롯을 확보한 후 수행합니다.
* 🧟 **프로세스 정리**: 학습이 비정상 종료된 경우, `nvidia-smi`로 VRAM에 남아있는 '좀비 프로세스'를 찾아 종료하세요.
* 🔄 **중단 후 재개(Resume)**: 예약 시간 종료에 대비해, 코드는 항상 최신 체크포인트를 저장하고 이어서 학습할 수 있도록 작성합니다.

### 5. smolVLA 및 Jetson 배포 특화 가이드

학습부터 기기 배포까지의 전체 파이프라인은 아래와 같습니다

#### 🚀 전체 배포 파이프라인

> **Server (학습)** → **QLoRA Fine-tuning** → **AWQ 양자화** → **Jetson으로 전송** → **TensorRT 변환** → **On-device 추론 서비스**

#### 상세 단계 안내

1. **Server (학습)**: 24GB VRAM 한계를 극복하기 위해 `BitsAndBytes` 라이브러리를 활용한 **QLoRA(4-bit) Fine-tuning**을 진행합니다.
2. **AWQ 양자화**: 학습이 완료된 가중치는 Jetson의 메모리 효율과 속도를 위해 `AutoAWQ` 라이브러리로 양자화합니다.
3. **Jetson으로 전송**: `scp` 또는 `sftp` 명령어를 사용하여 서버의 결과물을 Jetson Orin Nano 기기로 전송합니다.
4. **TensorRT 변환**: Jetson 기기 내부에서 NVIDIA의 가속 라이브러리인 **TensorRT** 포맷으로 최종 변환하여 하드웨어 성능을 극대화합니다.
5. **On-device 추론**: 변환된 모델을 바탕으로 실제 젯슨 기기에서 실시간 추론 서비스를 실행합니다.

### 6. 서버 데이터 보안 및 권한 관리

서버 내 데이터 보호와 조별 독립적인 환경을 위해 아래 권한 관리 규칙을 숙지하세요.

#### A. 소유권 및 접근 권한

* **소유권 변경 (`chown`)**: `/data2` 내 본인 폴더의 소유자가 본인인지 확인하세요. (예: `sudo chown -R student1:student1 /data2/student1`)
* **권한 설정 (`chmod`)**: 타 조의 접근을 차단하기 위해 본인 폴더는 `755`권한을 권장합니다.\
  (예: `chmod 755 data2/student1`)

| 권한 숫자   | 설명                      | 비고                 |
| ------- | ----------------------- | ------------------ |
| **700** | 나만 읽기/쓰기/실행 가능          | 조별 폴더 격리용          |
| **755** | 나는 모두 가능, 타인은 읽기/실행만 가능 | 데이터셋/스크립트용         |
| **777** | **누구나 삭제/수정 가능**        | **\[절대 금지]** 보안 위험 |

#### B. 학생 주의사항

**보안 및 경로 관리**

1. **chmod -R 777 금지**: 권한 에러 해결을 위해 777을 사용하는 것은 서버 전체의 보안을 위협합니다. 에러 발생 시 관리자에게 문의하세요.
2.  **심볼릭 링크 활용**: `/home` 용량 부족 시 `/data2`를 링크하여 편리하게 접근할 수 있습니다.

    ```
    # /data2의 본인 폴더를 home으로 연결
    ln -s /data2/student1 /home/student1/my_data
    ```

### 7. 필수 명령어 (Cheat Sheet)

| 상황              | 명령어                             |
| --------------- | ------------------------------- |
| **GPU 상태 확인**   | `nvtop` (강력 추천) 또는 `nvidia-smi` |
| **디스크 용량 확인**   | `df -h`                         |
| **내 pip 경로 확인** | `which pip`                     |
| **프로세스 강제 종료**  | `kill -9 [PID]`                 |
| **폴더 용량 확인**    | `du -sh *`                      |

{% hint style="info" %}
위 규칙을 어길 경우(특히 `/home` 디렉토리 용량 초과나 `sudo pip` 사용), 시스템 안정성을 위해 **가상환경이나 데이터가 삭제**될 수 있음을 유의하시기 바랍니다.
{% endhint %}
