---
description: Ubuntu 22.04 LTS 설치 직후 필수 유틸리티와 개발 도구를 세팅합니다.
metaLinks:
  alternates:
    - https://app.gitbook.com/s/yE16Xb3IemPxJWydtPOj/getting-started/quickstart
---

# 🐧 \[1] Linux 개발 환경 구축

### 1. Ubuntu 설치

{% hint style="info" %}
설치 언어는 English(US), Ubuntu 22.04 LTS인지 확인.
{% endhint %}

1. **ISO 다운로드:** [Ubuntu 22.04 LTS 공식 홈페이지](https://releases.ubuntu.com/jammy/)에서 ISO 파일을 다운로드합니다.
2. **부팅 디스크 제작:** `Rufus` 또는 `BalenaEtcher`를 사용하여 USB 부팅 디스크를 만듭니다.
3. **BIOS 설정:** 컴퓨터 재부팅 후 BIOS(F2 또는 Del 키)에 진입하여 부팅 순서를 USB 최우선으로 변경하고 `Secure Boot`를 해제(Disable)합니다.
4. **설치 진행:** 'Install Ubuntu'를 선택하고 안내에 따라 설치를 완료합니다.

### 2. 필수 패키지 설치 & 설정 (After Install)

설치가 완료되면 터미널(`Ctrl`+`Alt`+`T`)을 열고 아래 순서대로 세팅을 진행합니다.

#### 🖥️ 터미네이터 (Terminator)

기본 터미널보다 화면 분할이 자유로워 ROS 및 시뮬레이션 실행 시 필수적인 도구

```bash
sudo apt update
sudo apt install terminator -y
```

> **Tip:** 설치 후 `Win` 키를 누르고 `Terminator`를 검색해서 실행하세요. 우클릭 후 `Split Horizontally`(가로 분할) 등을 사용할 수 있습니다.

#### 🌐 구글 크롬 (Chrome)

```bash
# wget이 없을 경우 설치
sudo apt install -y wget

# 크롬 설치 파일 다운로드
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# 설치 진행
sudo apt install -y ./google-chrome-stable_current_amd64.deb
```

#### ⌨️ 한글 입력기 설정 (Fcitx5)

**1. 패키지 설치**

```bash
sudo apt install fcitx5 fcitx5-hangul -y
```

**2. 언어 지원 설정 (Setting)**

* `Setting` → `Region & Language` -> `Manage Installed Languages` 선택
* **Keyboard input method system** 항목을 `IBus`에서 `Fcitx 5`로 변경

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-04 20-58-34.png" alt=""><figcaption></figcaption></figure>

**3. 재부팅 (Reboot)** : 설정 적용을 위해 반드시 시스템을 재부팅 합니다.

```shellscript
reboot
```

**4. 한글 키 추가 및 Gnome Tweaks** : 재부팅 후 터미널을 열고 다음 과정을 수행합니다.

```shellscript
# Gnome Tweaks 설치 (UI 상세 설정 도구)
sudo apt install gnome-tweaks -y
```

**5. 한글 키 구성 (Tweaks Setting)**

* `Tweaks 실행` -> `Keyboard & Mouse` -> `Additional Layout Options`
* `Korean Hangul/Hanja Key` -> `Make right Alt a Hangul Key`

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-04 21-05-54.png" alt=""><figcaption></figcaption></figure>

**6. 입력기 구성 (Final Setting)**

* 우측 상단에 키보드 모양 클릭 -> `Configure`

<figure><img src="../.gitbook/assets/Screenshot from 2025-12-04 21-02-31.png" alt=""><figcaption></figcaption></figure>

* **Add Input Method**에서 `Hangul`을 검색하여 왼쪽 리스트(Current Input Method)에 추가합니다.
* `Global Options` 탭에서 **Trigger Input Method** (한영 전환 키)가 `Hangul` 키나 `Shift+Space`로 되어 있는지 확인합니다.

#### 💻 VS Code (Visual Studio Code) 설치

가장 대중적인 코드 에디터인 VS Code를 설치합니다.

```shellscript
# 1. 패키지 리스트 업데이트 및 필수 의존성 설치
sudo apt update
sudo apt upgrade -y
sudo apt install -y software-properties-common apt-transport-https wget

# 2. Microsoft GPG 키 다운로드 및 등록
wget -O- [https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc) | sudo gpg --dearmor | sudo tee /usr/share/keyrings/vscode.gpg

# 3. VS Code 저장소 추가
echo deb [arch=amd64 signed-by=/usr/share/keyrings/vscode.gpg] [https://packages.microsoft.com/repos/vscode](https://packages.microsoft.com/repos/vscode) stable main | sudo tee /etc/apt/sources.list.d/vscode.list

# 4. VS Code 설치
sudo apt update
sudo apt install code
```

설치가 완료되면 터미널에 `code`를 입력하거나 GUI를 통해실행할 수 있습니다.

> **🎉 준비 완료!** 위 과정이 모두 끝났다면 기본적인 개발 준비가 완료되었습니다. 다음 페이지에서는 **Miniconda 설치 및 가상환경 설정**을 진행하겠습니다.
