---
description: 로봇/AI 개발을 시작하기 전, 가장 중요한 기초 공사 단계입니다.
layout:
  width: default
  title:
    visible: true
  description:
    visible: true
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
    - >-
      https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/getting-started/publish-your-docs
---

# 🐧 \[2] Miniconda & VSCODE 세팅

### 1. 가상환경(Virtual Environment)의 필요성

#### 🏗️ "프로젝트별 독립적인 개발 환경 구축"

로봇 및 AI 연구를 수행하다 보면, 프로젝트마다 요구하는 **Python 버전**과 **라이브러리(PyTorch, TensorFlow 등)의 버전**이 서로 다른 경우가 빈번합니다.

예를 들어:

* **VLA 로봇 제어:** 최신 Python 3.10 및 PyTorch 2.1 환경 필요
* **기존 레거시 코드:** Python 3.7 및 TensorFlow 1.15 환경 필요

만약 이 모든 패키지를 하나의 로컬 컴퓨터(System Python)에 설치하게 되면, 라이브러리 간의 버전 충돌(Dependency Conflict)이 발생하여 코드가 정상적으로 실행되지 않는, 이른바 "의존성 지옥(Dependency Hell)"을 경험하게 됩니다.

**Miniconda**는 각 프로젝트마다 **완전히 격리된 가상 환경**을 생성하여, 서로 다른 버전의 라이브러리들이 충돌 없이 공존할 수 있도록 해주는 필수적인 도구입니다.

### 2. Miniconda 설치하기 (Linux/WSL)

터미널(Terminal)을 열고 아래 명령어들을 **한 줄씩 복사해서 실행**하세요.

**1) 설치 파일 다운로드**

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

**2) 설치 스크립트 실행**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

💡 **Tip:** 만약 설치 도중 `Proceed with initialization? [yes|no]` 질문이 나오면 `yes`를 입력하고 엔터를 누르세요. \
터미널을 켤 때마다 자동으로 `conda` 환경을 잡아주어 편리합니다.

**3) 터미널 초기화** : 설치가 끝난 후, 터미널을 껐다 켜거나 아래 명령어를 입력해야 `conda` 명령어가 인식됩니다.

```bash
source ~/.bashrc
```

> **확인:** 터미널 프롬프트 왼쪽에 `(base)`라는 글자가 생겼다면 성공입니다!

### 3. 로봇 실습용 가상환경 만들기

이제 이번 실습에서 사용할 가상환경을 만들어 봅시다. 이름은 `mujoco`으로 하겠습니다.

**1) 가상환경 생성 (Python 3.10 버전)**

```bash
conda create -n mujoco python=3.11 -y
```

> **⚠️ 트러블슈팅: 약관 동의 에러가 발생하는 경우**
>
> 만약 `conda create` 실행 시 `CondaToSNonInteractiveError` 에러가 뜬다면, 아나콘다 이용 약관 동의가 필요해서 그렇습니다. 아래 명령어를 입력하여 동의를 완료해주세요.
>
> ```
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
> ```

**2) 가상환경 들어가기 (Activate)**

```bash
conda activate mujoco
```

> **확인:** 프롬프트가 `(base)` ➡️ `(mujoco)`으로 바뀌었나요? 이제부터 설치하는 모든 라이브러리는 이 가상환경에만 저장됩니다.

**3) 라이브러리 설치 해보기**

```bash
pip install numpy
```

**4) 가상환경 나오기 (Deactivate)**

```bash
conda deactivate
```

### 4. Cheatsheet: 자주 쓰는 명령어 모음

| 기능           | 명령어                          |
| ------------ | ---------------------------- |
| **환경 목록 보기** | `conda env list`             |
| **환경 활성화**   | `conda activate [환경이름]`      |
| **환경 비활성화**  | `conda deactivate`           |
| **환경 삭제**    | `conda env remove -n [환경이름]` |
| **패키지 설치**   | `pip install [패키지명]`         |

### 5. VS Code 개발 환경 완벽 세팅

가상환경을 만들었어도 VS Code에서 이를 인식하려면 **확장 프로그램** 설치와 **인터프리터 설정**이 필수입니다.

#### 1) 필수 확장 프로그램(Extensions) 설치

VS Code 좌측의 **테트리스 블록 모양 아이콘(Extensions, `Ctrl+Shift+X`)**&#xC744; 클릭하고 아래 검색어를 입력하여 설치하세요.

| _<mark style="color:$primary;">**Extension**</mark>_ | _**설명**_                          |
| ---------------------------------------------------- | --------------------------------- |
| **Python** (Microsoft)                               |  파이썬 개발을 위한 **필수** 플러그인입니다.       |
| **Jupyter** (Microsoft)                              | `.ipynb` 파일(주피터 노트북) 실행을 지원합니다.   |
| **Remote - SSH** (Microsoft)                         | 연구실 서버 등 원격 리눅스 환경에 접속할 때 필수입니다.  |
| **Error Lens**                                       | (선택) 에러가 발생한 줄에 즉시 내용을 보여줍니다.     |
| **Material Icon**                                    | (선택) 파일 아이콘을 예쁘게 바꿔주어 가독성을 높여줍니다. |

#### 2) 가상환경 인터프리터(Interpreter) 연결

확장 프로그램을 설치했다면, 이제 VS Code가 우리가 방금 만든 가상환경을 쓰도록  알려줘야 합니다.

1. VS Code에서 `F1` 키 또는 `Ctrl + Shift + P`를 눌러 검색창(Command Palette)을 엽니다.
2. 입력창에 `Python: Select Interpreter`를 검색하고 클릭합니다.
3. 목록에서 `mujoco`이라고 적힌 항목을 찾아 선택합니다.

> **🎉 축하합니다!** 이제 모든 개발 환경 준비가 끝났습니다.

### 6. 가상환경 ON/OFF 차이 체험하기

"정말 가상환경이 필요한가?"를 눈으로 직접 확인해봅시다. \
우리가 방금 설치한 `mujoco` 환경에는 `numpy`를 설치했지만, 기본 `base` 환경에는 아직 없습니다.

#### 1단계: 테스트 파일 만들기

VS Code에서 `test_env.py`라는 파일을 만들고 딱 두 줄만 적으세요.

```shellscript
import numpy as np
print("✅ 성공! 현재 Numpy 버전:", np.__version__)
```

#### 2단계: 가상환경 끄고 실행해보기 (Fail 예상 ❌)

VS Code 아래쪽 터미널(\`Ctrl + \`\`)을 열고, 먼저 가상환경을 끕니다.

```shellscript
conda deactivate
python test_env.py
```

> **결과:** `ModuleNotFoundError: No module named 'numpy'`&#x20;

#### 3단계: 가상환경 켜고 실행해보기 (Success ✅)

이제 우리가 만든 방으로 들어가서 다시 실행해봅시다.

```shellscript
conda activate mujoco
python test_env.py
```

> **결과:** `✅ 성공! 현재 Numpy 버전: 2.x.x` 메시지가 뜨나요? 이것이 바로 가상환경을 쓰는 이유입니다!&#x20;
