---
description: 리더와 팔로워 암의 관절 위치를 똑같이 맞춘 뒤 리더암을 움직여 팔로워 로봇을 실시간으로 원격 조종하는 과정
icon: hand-fist
---

# \[30] SO-101: Calibrate & Teleop

## 조립 예시

#### 팔로워

<figure><img src="../.gitbook/assets/image (51).png" alt=""><figcaption></figcaption></figure>

#### 리더

<figure><img src="../.gitbook/assets/image (52).png" alt=""><figcaption></figcaption></figure>

#### 전체 사진

<figure><img src="../.gitbook/assets/image (53).png" alt=""><figcaption></figcaption></figure>

## 카메라 인덱스 확인

{% embed url="https://huggingface.co/docs/lerobot/cameras" %}

1. **아래 코드를 실행하여 연결된 모든 카메라의 인덱스를 확인한다.**

```bash
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
```

**출력 예시**

```bash
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ /dev/video0
  Type: OpenCV
  Id: /dev/video0
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
Camera #1:
  Name: OpenCV Camera @ /dev/video2
  Type: OpenCV
  Id: /dev/video2
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
Camera #2:
  Name: OpenCV Camera @ /dev/video4
  Type: OpenCV
  Id: /dev/video4
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
```

2. `lerobot/outputs/captured_images` 에서 생성된 카메라 이미지와 인덱스를 매칭해본다.

**예시:**

* /dev/video0 : 탑뷰 카메라
* /dev/video2: 팔로워암 카메라
* /dev/video4: 랩탑 카메라

## Calibrate

_**미들 포지션**_

<figure><img src="../.gitbook/assets/image (54).png" alt=""><figcaption></figcaption></figure>

### 팔로워암: Calibrate

1. **아래 명령어를 입력한다.**

```bash
# 팔로워암
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_FOLLOWER \
    --robot.id=my_follower
```

2. `Move my_follower SO101Follower to the middle of its range of motion and press ENTER....` 출력이 나오면 미들 포지션으로 만들고 엔터
3. 모든 조인트를 최소/최대로 스트레칭 해준다
4. 다음과 같은 결과가 나오는지 확인

```bash
-------------------------------------------
-------------------------------------------
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1312 |   1949 |   2723
shoulder_lift   |    923 |   2087 |   3291
elbow_flex      |    840 |   2041 |   3046
wrist_flex      |    858 |   2097 |   3191
wrist_roll      |    125 |   1983 |   3922
gripper         |   2004 |   2040 |   3483

```

5. 생성된 calibration 파일 확인 `~/.cache//huggingface/lerobot/calibration/robots/so101_follower/my_follower.json`

### 리더암: Calibrate

1. **아래 명령어를 입력한다.**

```bash
# 리더암
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_LEADER \
    --teleop.id=my_leader 
```

2. `Move my_leader SO101Leader to the middle of its range of motion and press ENTER....` 출력이 나오면 미들 포지션으로 만들고 엔터
3. 모든 조인트를 최소/최대로 스트레칭 해준다
4. 다음과 같은 결과가 나오는지 확인

```bash
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1133 |   1939 |   2759
shoulder_lift   |    896 |    902 |   3260
elbow_flex      |    826 |   3114 |   3118
wrist_flex      |    857 |   2159 |   3178
wrist_roll      |    112 |   2046 |   3903
gripper         |   2032 |   2055 |   3234
```

5. 생성된 calibration 파일 확인 `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_leader.json`

## Teleoperation

{% embed url="https://huggingface.co/docs/lerobot/il_robots" %}

1. 아래 코드를 실행한다. (리더, 팔로워 포트 모두 입력)

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_FOLLOWER \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_LEADER \
    --teleop.id=my_leader
```

2. 리더암을 조작시 팔로우암의 움직임이 정상적인지 확인한다.
3. 카메라 인덱스를 활용해 **Teleoperate with cameras**

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_FOLLOWER \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_LEADER \
    --teleop.id=my_leader \
    --display_data=true
```

4. 리더암을 조작시 팔로우암의 움직임이 정상적인지 확인한다.

<figure><img src="../.gitbook/assets/image (55).png" alt=""><figcaption></figcaption></figure>

5. 카메라 초점 조절

<figure><img src="../.gitbook/assets/image (56).png" alt=""><figcaption></figcaption></figure>
