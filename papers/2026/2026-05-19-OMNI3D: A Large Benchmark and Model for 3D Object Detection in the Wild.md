# Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild

- **학회:** CVPR 2023
- **링크:**
  - Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Brazil_Omni3D_A_Large_Benchmark_and_Model_for_3D_Object_Detection_CVPR_2023_paper.pdf
  - arXiv: https://arxiv.org/abs/2207.10660
- **코드:** https://github.com/facebookresearch/omni3d
- **분야:** Image-based / Monocular 3D Object Detection, 3D Object Detection Benchmark, Multi-domain 3D Recognition

---

## 1. 요약

- 본 논문은 단일 RGB 이미지로부터 객체의 3D 위치, 크기, 회전을 예측하는 **image-based 3D object detection** 문제를 다룬다.
- 기존 3D 객체 검출 벤치마크는 KITTI, SUN RGB-D처럼 규모가 작고 특정 도메인에 편향되어 있어, 일반적인 3D 객체 인식 모델을 학습하기 어렵다.
- 이를 해결하기 위해 저자들은 기존 3D 데이터셋들을 통합하여 **OMNI3D**라는 대규모 3D 객체 검출 벤치마크를 구축했다.
- OMNI3D는 SUN RGB-D, ARKitScenes, Hypersim, Objectron, KITTI, nuScenes를 재가공하여 만든 데이터셋이며, 약 **234k 이미지**, **300만 개 이상의 3D box annotation**, **98개 카테고리**를 포함한다.
- 모델 측면에서는 Faster R-CNN을 확장한 **Cube R-CNN**을 제안한다.
- Cube R-CNN은 2D 객체를 검출한 뒤, 각 2D RoI에 대해 3D cuboid의 center, depth, size, rotation, uncertainty를 예측한다.
- 다양한 카메라 intrinsic을 가진 데이터셋을 함께 학습하기 위해 **virtual depth**를 도입한다.
- Virtual depth는 실제 metric depth를 공통 virtual camera 기준으로 변환하여 focal length 차이에 따른 scale-depth ambiguity를 줄인다.
- 또한 3D box IoU 계산을 빠르게 하기 위한 batched C++/CUDA 기반 **Fast IoU3D** 알고리즘을 구현한다.
- 실험 결과 Cube R-CNN은 OMNI3D, KITTI, SUN RGB-D 등에서 기존 방법보다 우수하거나 비슷한 성능을 보이며, OMNI3D pre-training은 작은 3D 데이터셋에서 low-shot 학습을 가속한다.

---

## 2. 핵심 기여

- **대규모 3D 객체 검출 벤치마크 OMNI3D 제안**
  - 기존 3D 데이터셋들을 하나의 통합된 camera coordinate system과 annotation format으로 재가공했다.
  - 기존 대표 벤치마크인 KITTI, SUN RGB-D보다 훨씬 큰 규모와 다양한 도메인, 카테고리, 카메라 intrinsic을 제공한다.

- **범용 단안 3D 객체 검출기 Cube R-CNN 제안**
  - Faster R-CNN에 3D cube head를 추가하여, 단일 RGB 이미지에서 2D box와 3D cuboid를 end-to-end로 예측한다.
  - 특정 도메인, 예를 들어 자율주행이나 실내 장면에 특화된 가정 없이 indoor/outdoor를 하나의 모델로 처리한다.

- **Virtual Depth 및 Fast IoU3D 도입**
  - Virtual depth는 서로 다른 focal length와 image scale로 인한 depth ambiguity를 줄이고, 3D detection에서도 scale augmentation을 가능하게 한다.
  - Fast IoU3D는 임의 회전 3D cuboid 간의 정확한 IoU를 빠르게 계산하여 대규모 OMNI3D 평가를 현실적으로 가능하게 한다.

---

## 3. 방법

### 입력

- 단일 RGB 이미지
- 카메라 intrinsic
  - focal length: `fx`, `fy`
  - principal point: `px`, `py`
- 학습 시 ground-truth annotation
  - 2D bounding box
  - 3D bounding box / cuboid
  - 3D center in camera coordinates
  - object dimensions: width, height, length
  - object-to-camera rotation matrix
  - category label

---

### 핵심 아이디어

#### 1) Faster R-CNN 기반 구조

Cube R-CNN은 Faster R-CNN을 기반으로 한다.

```text
RGB image
  → Backbone CNN + FPN
  → RPN
  → RoI feature
  → 2D box head: category + 2D box
  → cube head: 3D cuboid parameters
```

즉, 먼저 2D 객체 후보를 찾고, 각 RoI에 대해 3D box를 회귀한다.

---

#### 2) RPN objectness를 IoUness로 대체

기존 Faster R-CNN의 RPN은 region이 객체인지 아닌지를 예측하는 **objectness classifier**를 사용한다.

하지만 OMNI3D는 여러 데이터셋을 합친 benchmark이므로 모든 객체가 완전하게 라벨링되어 있다고 보장하기 어렵다. 이 경우 실제 객체가 있는 region이 background로 잘못 학습될 수 있다.

이를 완화하기 위해 Cube R-CNN은 objectness 대신 **IoUness regressor**를 사용한다.

```text
기존 RPN: region → object / background
Cube R-CNN: region → GT box와의 IoU 정도
```

RPN loss는 IoU가 높은 후보를 더 중요하게 학습하도록 구성된다.

---

#### 3) Cube Head

Cube head는 각 2D RoI feature로부터 3D cuboid를 정의하는 13개 파라미터를 예측한다.

```text
[u, v]       : RoI 기준 projected 3D center
z 또는 zv    : object center depth
[w, h, l]    : physical 3D box dimension
p ∈ R^6      : 6D allocentric rotation
μ            : 3D uncertainty
```

3D center는 projected center와 depth, camera intrinsic을 이용해 camera coordinate로 back-projection한다.

```text
2D projected center + depth + camera intrinsic
→ 3D center in camera coordinates
```

Box dimension은 category-specific 평균 크기에 대한 log-normalized residual로 예측한다. Rotation은 6D continuous representation으로 예측하고, allocentric rotation을 egocentric rotation으로 변환하여 최종 3D box를 구성한다.

---

#### 4) 3D Cuboid 생성

Cube head가 예측한 center, size, rotation을 이용해 unit cube를 변환한다.

```text
unit cube
  → scale: width / height / length
  → rotate: 3D rotation matrix
  → translate: 3D center
  → final 3D cuboid
```

최종 cuboid는 8개 corner로 표현된다.

---

#### 5) Virtual Depth

OMNI3D는 여러 카메라와 여러 데이터셋을 합친 것이므로 focal length와 image resolution이 크게 다르다. 같은 실제 depth에 있는 객체라도 focal length가 다르면 이미지상 크기가 달라져 scale-depth ambiguity가 커진다.

이를 해결하기 위해 모델은 metric depth `z`를 직접 예측하지 않고, 공통 virtual camera 기준으로 정규화된 **virtual depth `zv`**를 예측한다.

```text
zv = z × (fv / f) × (H / Hv)
```

- `z`: 실제 metric depth
- `f`: 실제 camera focal length
- `H`: 실제 image height
- `fv`: virtual focal length
- `Hv`: virtual image height

Inference에서는 예측된 `zv`를 실제 camera intrinsic을 사용해 다시 metric depth `z`로 복원한다.

Virtual depth의 효과는 두 가지이다.

- 서로 다른 camera intrinsic을 가진 이미지를 하나의 모델에서 안정적으로 학습할 수 있다.
- image resizing / scale augmentation을 3D detection에서도 사용할 수 있다.

---

#### 6) 3D Loss

3D loss는 크게 두 종류로 구성된다.

첫째, 예측된 3D box와 GT 3D box의 8개 corner를 point cloud처럼 보고 chamfer loss를 계산하는 **entangled full box loss**를 사용한다.

둘째, center, depth, size, rotation을 각각 분리해서 학습하는 **disentangled loss**를 사용한다. 예를 들어 center loss를 계산할 때는 center만 예측값을 사용하고, depth/size/rotation은 GT 값을 사용한다.

```text
L3D = center loss
    + depth loss
    + size loss
    + rotation loss
    + full cuboid loss
```

최종 학습 loss는 다음 요소들을 포함한다.

```text
L = RPN loss
  + 2D detection loss
  + uncertainty-weighted 3D loss
  + uncertainty penalty
```

3D uncertainty `μ`는 학습 시 어려운 sample의 3D loss를 조절하고, inference 시 최종 3D detection confidence에 반영된다.

---

#### 7) Fast IoU3D

기존 KITTI 방식은 3D box를 ground plane에 투영하여 top-view intersection과 height를 곱하는 근사 방식을 사용한다. 이 방식은 객체가 지면 위에 있지 않거나 pitch/roll이 있는 경우 부정확하다.

논문은 cuboid를 mesh로 표현하고 face intersection을 계산하는 방식의 정확한 IoU3D 알고리즘을 구현한다.

특징은 다음과 같다.

- arbitrary oriented 3D cuboid 지원
- batched computation 지원
- C++ / CUDA 구현
- Objectron 구현보다 C++ 기준 90배, CUDA 기준 450배 빠름

---

### 출력

Cube R-CNN의 최종 출력은 이미지 내 각 객체에 대해 다음 정보를 포함한다.

- 2D bounding box
- object category
- 3D center in camera coordinates
- metric depth
- 3D box dimensions: width, height, length
- 3D rotation matrix
- 8-corner 3D cuboid
- 3D uncertainty
- 최종 detection confidence score

---

## 4. 메모

- 이 논문의 핵심은 **새로운 네트워크 구조 자체보다, 대규모 multi-domain 3D benchmark와 이를 안정적으로 학습하기 위한 설계**에 있다.
- OMNI3D는 새로 모든 3D bbox를 수작업 라벨링한 데이터셋이라기보다는, 기존 데이터셋의 3D annotation을 통합 좌표계와 통합 포맷으로 재가공한 benchmark에 가깝다.
- 3D bbox GT는 각 source dataset의 특성에 따라 RGB-D, LiDAR, synthetic mesh, 기존 3D box annotation 등을 기반으로 한다.
- 실내 집 환경에서 3D bbox GT를 만들려면 OMNI3D 자체보다는 SUN RGB-D나 ARKitScenes 방식이 더 직접적인 참고가 된다.
- 현실적인 3D bbox annotation pipeline은 다음과 같다.

```text
RGB-D / LiDAR / SLAM / reconstruction으로 scene point cloud 또는 mesh 생성
→ gravity alignment
→ 사람이 3D viewer에서 object-level oriented cuboid annotation
→ 3D center, dimensions, rotation, 8 corners 저장
→ 각 RGB frame의 camera pose + intrinsic으로 2D projection
→ image별 2D bbox + 3D bbox pair 생성
```

- Cube R-CNN은 camera intrinsic이 알려져 있다는 가정에 의존한다. 실제 custom data에 적용하려면 RGB camera calibration이 중요하다.
- Virtual depth 아이디어는 서로 다른 카메라나 서로 다른 focal length를 가진 데이터를 섞어 학습할 때 유용하다. 사용자가 여러 센서 또는 여러 촬영 환경에서 3D object detection GT를 만들려는 경우 특히 중요하다.
- 집안 물체 3D volume 검출 목적에서는 Cube R-CNN을 바로 사용하는 것보다, reconstruction 기반 3D bbox annotation으로 pseudo-GT를 만들고, 이를 image-based 3D detector 학습에 사용하는 방향이 현실적이다.


---

## 5. 적용 포인트
 - Omni3D DB의 구조 파악
 - 3D Rotation 표현 방법
 - Virtual depth 표현방법
 - disentangled loss 방법 
 - 큐보이드 IOU 가속 계산법
