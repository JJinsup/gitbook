---
icon: landmark-magnifying-glass
---

# Git

### Git 환경 설정

_**1️⃣ Git 연결 설정**_

```bash
# ssh key 생성
ssh-keygen -t ed25519 -C "limsk519@kookmin.ac.kr"
# 공개 키 복사
cat ~/.ssh/id_ed25519.pub
```

_**GitHub 웹사이트에 등록**_

* **GitHub 로그인** 후 우측 상단 프로필 클릭 -> **Settings**
* 좌측 메뉴에서 **\[SSH and GPG keys]** 클릭
* **\[New SSH key]** (녹색 버튼) 클릭
* **Title**: 식별하기 좋은 이름 (예: `Server`)
* **Key**: 아까 복사한 내용을 그대로 붙여넣기
* **Add SSH key** 버튼 클릭

```bash
# ssh PORT 변경 아래 내용을 추가
nano ~/.ssh/config

###################################
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
###################################

# 연결 테스트 및 Git 유저 설정
ssh -T git@github.com
git config --global user.name "JJinsup"
git config --global user.email "limjs519@gmail.com"
```

**1️⃣ `.git` 지우고, 그냥 “내 저장소”로 가져오기**

```bash
# 이미 Clone 되어 있는 상황
git remote rename origin upstream

rm -rf .git

git init -b main  # git 버전에 따라 안 되면 git init 후 아래에서 branch 이름 바꿔도 됨
```

**2️⃣ GitHub 웹사이트에서 저장소(Repository) 만들기**

**3️⃣ 서버에서 새 origin 연결 + push**

```bash
# 새 git 저장소 초기화 (main 브랜치로)
git init -b main  # git 버전에 따라 안 되면 git init 후 아래에서 branch 이름 바꿔도 됨

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit based on LycheeAI SO-ARM100/101 (BSD-3-Clause)"

# 네 GitHub 원격 추가
git remote add origin <https://github.com/jjinsup/so_arm_101_isaac.git>

# 푸시
git push -u origin main
```

{% hint style="info" %}
```
# 주소를 SSH 형식으로 변경 (ssh방식으로 푸시 실패할 경우)
git remote set-url origin git@github.com:<유저>/레포.git
```
{% endhint %}
