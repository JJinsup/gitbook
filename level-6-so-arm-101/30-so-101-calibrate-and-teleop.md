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

### 리더암: Calibrate

