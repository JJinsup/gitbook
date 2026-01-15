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

# Remote URL을 SSH로 변경
# 현재 주소가 HTTPS인지 확인 (출력에 https://... 가 보일 겁니다)
git remote -v

# 주소를 SSH 형식으로 변경
git remote set-url origin git@github.com:JJinsup/rfSoC-Book.git
```

**2️⃣ GitHub 웹사이트에서 저장소(Repository) 만들기**

**3️⃣ 로컬(서버)과 원격(GitHub) 연결하기**

```bash
# 폴더 만들고 권한 설정 및 확인
sudo mkdir /data1/js
sudo chown -R lab602:lab602 /data1/js
ls -ld /data1/js

# 바로가기(심볼릭 링크) 만들기
cd
ln -s /data1/js js

# 깃허브 저장소 불러오기
git clone git@github.com:JJinsup/tf-learning.git
cd tf-learning
```

#### 이미 존재하는 폴더 Git init해서 연동하는법

```jsx
cd /data/jinsup/js_mujoco
git init
nano .gitignore   # (optional but recommended)
git add .
git commit -m "Initial commit"
git remote add origin <https://github.com/ID/REPO.git>
git branch -M main
git push -u origin main
```
