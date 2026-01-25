---
description: >-
  이 문서는 Ubuntu 22.04 기반의 JetPack 6.0 설치 과정을 다룹니다. LeRobot 및 smolVLA 구동을 위해 반드시 이
  버전을 준수해야 합니다.
icon: landmark-magnifying-glass
---

# Jetson Orin Nano JetPack 6.0 설치 가이드

**For Jetson Orin:**

* Jetson JetPack 6.0 and 6.1, not support 6.1
* Python 3.10
* Torch 2.3+

아래 사이트들을 참고해서 제트팩 6.0으로 설치합니다.

{% embed url="https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit" %}

{% embed url="https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/FlashingSupport.html" %}

{% embed url="https://operationcoding.tistory.com/217" %}

#### SSD Install은 아래 영상을 참고하세요.

{% embed url="https://www.youtube.com/watch?v=BaRdpSXU6EM" %}

#### 1. 준비물

* microSD 카드: 최소 64GB 이상 (AI 모델 용량을 고려해 128GB 권장, UHS-I 등급 이상 필수).
* SD 카드 리더기: PC 연결용.
* 설치 프로그램: [Balena Etcher](https://www.balena.io/etcher/) 및 [SD Card Formatter](https://www.sdcard.org/downloads/formatter/).
* JetPack 이미지: [NVIDIA Jetson Orin Nano용 SD 카드 이미지 (JetPack 6.0)](https://developer.nvidia.com/embedded/jetpack).

{% embed url="https://developer.nvidia.com/embedded/jetpack-archive" %}

#### 2. SD 카드 플래싱

가장 먼저 SD 카드에 운영체제 이미지를 구워야 합니다.

1. SD 카드 포맷: SD Card Formatter를 실행하여 'Quick Format'을 진행합니다.
2. Balena Etcher 실행:
   * Flash from file: 다운로드한 JetPack 6.0 이미지 파일(.zip)을 선택합니다.
   * Select target: 포맷한 microSD 카드를 선택합니다.
   * Flash!: 버튼을 눌러 쓰기를 시작합니다. 완료 후 'Flash Complete!' 메시지가 뜨면 카드를 분리합니다.

#### 3. 하드웨어 연결 및 부팅

1. SD 카드 삽입: Jetson Orin Nano 모듈 하단의 슬롯에 microSD 카드를 끝까지 밀어 넣습니다.
2. 주변기기 연결: 모니터(HDMI/DP), 키보드, 마우스를 연결합니다.
3. 전원 연결: USB-C 또는 DC 잭을 통해 전원을 연결하면 자동으로 부팅이 시작됩니다.
4. **부팅 실패 시 SDK 매니저를 이용한 펌웨어 업데이트가 필요할 수도 있습니다 -> SDK 매니저를 이용한 펌웨어(QSPI) 업데이트 가이드**로 이동

#### 4. 초기 OS 설정 (Ubuntu Setup)

부팅 후 화면의 안내에 따라 시스템 설정을 진행합니다.

1. 약관 동의: 'I accept the terms...' 체크 후 Continue.
2. 언어 및 키보드: 영어(English) 권장 (코딩 및 경로 오류 방지용).
3. 네트워크: Wi-Fi 또는 유선 랜 연결.
4. 사용자 설정 (중요):
   * Name / Username / Password: 학생들이 기억하기 쉬운 간단한 정보로 설정합니다.
5. 설치 완료: 설정이 끝나면 재부팅되며 Ubuntu 데스크탑 화면이 나타납니다.

#### 5. 실습을 위한 필수 라이브러리 설치

이미지에는 기본 OS만 포함되어 있으므로, CUDA와 TensorRT 같은 AI 구성 요소를 별도로 설치해야 합니다.

1. 터미널 실행: `Ctrl + Alt + T`를 누릅니다.
2.  JetPack SDK 설치: 아래 명령어를 순서대로 입력합니다.

    ```
    sudo apt update
    sudo apt install nvidia-jetpack
    ```

    * 이 작업은 용량이 크고 시간이 다소 소요됩니다.
3.  환경 변수 추가: CUDA 명령어를 어디서든 사용할 수 있게 설정합니다.

    ```
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```
4.  Install Miniconda: For Jetson:

    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    chmod +x Miniconda3-latest-Linux-aarch64.sh
    ./Miniconda3-latest-Linux-aarch64.sh
    source ~/.bashrc
    ```

&#x20;**PyTorch 설치가 따로 필요한 이유**

{% embed url="https://github.com/Seeed-Projects/reComputer-Jetson-for-Beginners/tree/main/3-Basic-Tools-and-Getting-Started/3.5-Pytorch" %}

* GPU 가속 지원: NVIDIA는 젯슨의 GPU 아키텍처에 최적화된 PyTorch 빌드(.whl 파일)를 별도로 제공합니다.
* LeRobot 주의사항: `pip install -e ".[feetech]"` 명령어를 실행할 때, `pip`이 기존에 설치된 GPU용 PyTorch를 삭제하고 CPU 전용 버전을 덮어씌울 수 있어 주의가 필요합니다.

***

**단계별 설치 명령어**

**A. 필수 의존성 라이브러리 설치**

```
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```

**B. PyTorch 2.3.0 (JetPack 6.0용) 설치**

NVIDIA에서 제공하는 JetPack 6.0용 PyTorch 2.3.0 버전을 설치합니다.

```
# PyTorch 다운로드 및 설치
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2ad6.nv24.03-cp310-cp310-linux_aarch64.whl
pip3 install --no-cache $TORCH_INSTALL
```

**C. Torchvision 0.18.0 설치**

설치한 PyTorch 버전에 맞는 Torchvision을 소스 빌드 또는 설치합니다.

```
# Torchvision 소스 다운로드
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
cd torchvision
# GPU 가속을 포함하여 설치
export BUILD_VERSION=0.18.0
python3 setup.py install --user
```

***

설치가 끝난 후 반드시 GPU 인식이 되는지 확인해야 합니다.

1. 터미널에 `python3`를 입력하여 파이썬 인터프리터를 실행합니다.
2.  아래 코드를 입력하여 `True`가 나오는지 확인합니다.

    ```
    import torch
    print(torch.cuda.is_available())
    # 결과가 True이면 성공, False이면 재설치 필요
    ```

***

💡 팁

* LeRobot 설치 후 재점검: LeRobot 라이브러리를 설치한 직후에 만약 결과가 `False`로 변했다면 PyTorch만 다시 설치해야 합니다.
* Numpy 버전 호환성: Torchvision과 Numpy 버전이 맞지 않아 오류가 날 수 있습니다. 이 경우 `pip install numpy==1.26.0`으로 버전을 맞춰주면 해결됩니다.

#### 6. 가상 메모리(Swap) 설정

Jetson Orin Nano의 메모리(8GB/4GB) 부족으로 인해 시스템이 멈추거나 PyTorch 빌드가 실패하는 것을 방지합니다.

* 목표: SD 카드의 일부를 메모리처럼 사용하여 물리적 메모리 한계를 극복합니다.

```
# 8GB 크기의 스왑 파일 생성
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 재부팅 시에도 유지되도록 설정
echo '/swapfile sw sw 0 0' | sudo tee -a /etc/fstab
```

***

#### 7. 환경 최적화 및 마무리

라이브러리 간의 충돌을 방지하고 로봇 제어 시 실시간 성능을 최대화합니다.

*   Numpy 버전 고정: Torchvision 빌드 후 버전이 꼬이는 것을 방지하기 위해 다시 한번 실행합니다.

    ```
    pip install numpy==1.26.0
    ```
*   전원 모드 최대화: 로봇의 추론 속도(FPS)를 높이기 위해 하드웨어 성능을 최대(MAX)로 끌어올립니다.

    ```
    # 전원 모드를 MAX로 설정 (팬 소음이 커질 수 있음)
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```
*   통신 권한 설정: 로봇 암(SO-101) 제어 보드에 접근할 수 있도록 USB 포트 권한을 개방합니다.

    ```
    sudo chmod 666 /dev/ttyACM*
    ```

## **SDK 매니저를 이용한 펌웨어(QSPI) 업데이트 가이드**

{% embed url="https://www.youtube.com/watch?v=Ucg5Zqm9ZMk" %}

#### 0. NVIDIA SDK Manager Install

{% embed url="https://developer.nvidia.com/sdk-manager" %}

* `Ubuntu용 .deb` 패키지 파일을 다운로드합니다.

```bash
# 다운로드 폴더로 이동
cd ~/Downloads

# 설치 명령어 실행 (파일명은 실제 다운로드된 파일로 변경)
sudo apt install ./sdkmanager_2.x.x-xxxx_amd64.deb
```

#### 1. 호스트 PC 세팅 및 Jetson 연결

* 호스트 PC: Ubuntu 22.04(또는 20.04)가 설치된 PC에서 NVIDIA SDK Manager를 실행합니다.
* 점퍼 연결: Jetson의 전원을 끈 상태에서 9번과 10번 핀을 점퍼로 연결(쇼트)합니다.
* 케이블 연결: Jetson 전면의 USB-C 포트와 호스트 PC를 데이터 케이블로 연결한 뒤 Jetson의 전원을 켭니다.

#### 2. SDK 매니저 단계별 설정 (Step 01 \~ 03)

STEP 01: 보드 및 버전 선택

* Target Hardware: `Jetson Orin Nano` 선택.
* Target Operating System: `JetPack 6.0` 선택.
* 추가 구성 요소: `Host Machine` 항목은 체크 해제하여 시간을 단축합니다.

STEP 02: 설치 항목 선택 (매우 중요)

* Jetson OS: 이 항목만 체크합니다. (이 안에 펌웨어 업데이트 파일이 포함되어 있습니다.)
* Jetson SDK Components: 이 항목은 체크 해제합니다. (CUDA, TensorRT 등은 나중에 SD 카드 부팅 후 Jetson에서 직접 설치하는 것이 훨씬 빠르고 간편합니다.)

STEP 03: 플래싱 설정 팝업 `Install` 버튼을 누르면 설치 옵션을 묻는 팝업이 뜹니다.

* Manual Setup: 선택.
* Storage Device: SD 카드 이미지를 따로 구울 예정이라도, 여기서는 `Internal eMMC/SD`를 선택해야 보드 내부의 QSPI 펌웨어 쓰기 작업이 진행됩니다.
* Flash 버튼 클릭: 이제 호스트 PC가 Jetson의 내부 펌웨어를 최신 버전으로 업데이트하기 시작합니다.

#### 3. 마무리 및 부팅 확인

1. 플래싱 완료: SDK 매니저에서 "Flash 완료" 메시지가 뜨면 Jetson의 전원을 끕니다.
2. 점퍼 제거: 연결했던 9-10번 점퍼를 반드시 제거합니다. (제거하지 않으면 계속 복구 모드로만 부팅됩니다.)
3. SD 카드 삽입: 미리 구워둔 JetPack 6.0 SD 카드를 슬롯에 꽂습니다.
4. 부팅: 전원을 켜면 이제 보드가 SD 카드를 인식하고 정상적인 Ubuntu 설정 화면이 나타납니다.
