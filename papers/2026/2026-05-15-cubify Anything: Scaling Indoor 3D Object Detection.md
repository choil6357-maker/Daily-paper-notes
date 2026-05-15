# Cubify Anything: Scaling Indoor 3D Object Detection

- **학회:** CVPR 2025
- **링크:** https://arxiv.org/abs/2412.04458  
  - CVF PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Lazarow_Cubify_Anything_Scaling_Indoor_3D_Object_Detection_CVPR_2025_paper.pdf
- **코드:** https://github.com/apple/ml-cubifyanything
- **분야:** Indoor 3D Object Detection, RGB-D 3D Detection, Image-based 3D Box Prediction, 3D Dataset Construction

---

## 1. 요약
- 본 논문은 상용 handheld device에서 얻은 **단일 RGB(-D) 프레임**을 입력으로 실내 객체의 **3D bounding box**를 예측하는 문제를 다룬다.
- 기존 실내 3D detection dataset은 주로 noisy depth/pose 기반 point cloud나 mesh에 annotation되어 있어, annotation bias와 3D reconstruction noise가 모델 학습에 얽히는 문제가 있었다.
- 이를 해결하기 위해 저자들은 **CA-1M, Cubify-Anything 1M** 데이터셋을 제안한다.
- CA-1M은 ARKitScenes의 원본 handheld RGB-D capture와 FARO laser scan registration을 활용하되, 객체 annotation은 noisy ARKit mesh가 아니라 **고정밀 FARO laser scan** 위에서 수행한다.
- CA-1M은 1,000개 이상의 실내 scene, 3,500개 이상의 handheld capture, 439K개 이상의 unique 3D object, 약 13M training frame을 포함한다.
- 데이터셋은 cabinet, table, bed 같은 room-defining object뿐 아니라 작은 생활 객체까지 **class-agnostic하게 exhaustive annotation**하는 것을 목표로 한다.
- 모델 측면에서는 **CuTR, Cubify Transformer**를 제안한다.
- CuTR은 point cloud나 voxel을 입력으로 사용하지 않고, RGB 또는 RGB-D image feature에서 **2D box와 3D box를 직접 예측**한다.
- RGB-D CuTR은 MultiMAE 기반 ViT backbone을 사용해 RGB와 affine-invariant depth token을 함께 encoding하고, RGB-only CuTR은 Depth-Anything 기반 initialization을 사용한다.
- 실험적으로 CA-1M에서 CuTR은 point-based method보다 높은 recall/precision을 보이며, noisy commodity LiDAR depth 환경에서 특히 강한 성능을 보인다.

---

## 2. 핵심 기여
- **CA-1M 데이터셋 구축**
  - ARKitScenes 기반 handheld RGB-D capture와 FARO laser scan registration을 활용.
  - 3D annotation은 handheld LiDAR/SLAM 기반 noisy mesh가 아니라 고정밀 FARO scan 위에서 수행.
  - 439K개 이상의 object를 class-agnostic하게 라벨링하고, 각 handheld frame으로 pixel-accurate하게 rendering하여 frame-level 2D/3D GT를 생성.

- **Image-level 3D object detection 문제 재정의**
  - 기존 실내 3D detection은 scene-level point cloud/mesh 입력에 치우쳐 있었음.
  - 본 논문은 단일 RGB(-D) frame에서 object-level 3D box를 예측하는 설정을 강조.
  - 2D detection과 더 유사한 image-level task로 3D object detection을 재정렬함.

- **CuTR 모델 제안**
  - point/voxel/sparse convolution 없이 ViT + DETR-style detector만으로 3D box를 예측.
  - 3D center를 직접 3D 좌표로 예측하지 않고, image plane의 projected center xy와 depth z를 예측한 뒤 camera intrinsic으로 backprojection.
  - 3D inductive bias가 약한 image-based model도 충분히 크고 정확한 데이터가 있으면 point-based model을 능가할 수 있음을 보임.

---

## 3. 방법

### 입력
- **RGB-D CuTR**
  - RGB image
  - optional metric depth map
  - camera intrinsic K
  - gravity-aligned coordinate transform
- **RGB-only CuTR**
  - RGB image
  - camera intrinsic K
  - gravity-aligned coordinate transform
- **CA-1M DB 생성 입력**
  - iPad Pro handheld RGB-D captures
  - ARKit pose/depth
  - FARO laser scan
  - FARO scan ↔ handheld capture registration

### 핵심 아이디어

#### 3.1 CA-1M DB 생성
- 기존 ARKitScenes는 handheld LiDAR와 on-device SLAM으로 만든 noisy mesh 위에 주로 큰 객체의 3D box를 annotation한다.
- CA-1M은 동일한 underlying capture를 사용하지만, 3D box annotation을 **FARO laser scan** 위에서 수행한다.
- Annotation tool은 FARO point cloud뿐 아니라 대응되는 RGB frame overlay와 supporting frames를 함께 제공한다.
- 투명/반사 객체처럼 laser scan이 불완전한 경우에도 RGB evidence를 함께 보며 annotation한다.
- Bootstrapped CuTR를 annotation loop에 넣어, annotator가 이미지에서 2D box를 그리면 이를 3D annotation 초기값으로 사용할 수 있게 한다.
- World-space 3D box annotation을 각 video frame으로 rendering하여, camera frustum과 occlusion을 반영한 frame-level 2D/3D GT를 생성한다.
- 결과적으로 annotation은 3D spatial reality를 반영하면서도 RGB image에 pixel-perfect하게 정렬된다.

#### 3.2 CuTR 모델 구조
- CuTR은 2D detector를 3D object detection으로 확장한 single-stage, single-scale Transformer detector이다.
- Backbone은 ViT 계열을 사용한다.
- RGB-D variant는 MultiMAE를 사용해 RGB token과 affine-invariant depth token을 jointly encode한다.
- RGB-only variant는 Depth-Anything 기반 ViT를 initialization으로 사용해 monocular geometry prior를 활용한다.
- Detector는 Plain DETR 스타일의 object query 기반 decoder를 사용한다.
- 각 query는 다음을 예측한다.
  - 2D bounding box
  - 3D center의 projected xy
  - 3D center depth z
  - 3D box dimensions: length, width, height
  - 3D box yaw orientation
- RGB-D variant에서는 metric depth의 평균 μ와 표준편차 σ를 저장한 뒤, normalized prediction을 metric scale로 복원한다.
  - z′ = σz + μ
  - (l′, w′, h′) = (σl, σw, σh)
- 예측된 projected xy와 depth z′는 camera intrinsic K를 사용해 3D center로 backprojection된다.
- Orientation은 gravity-aware setting에서 yaw만 예측하며, pitch/roll 예측은 future work로 남긴다.
- 3D box corner는 Chamfer loss로 supervise한다.
- Ground-truth assignment는 2D box prediction 기준 Hungarian matching으로 수행하며, CuTR은 NMS에 의존하지 않는다.

### 출력
- 각 입력 frame에 대해:
  - 2D bounding boxes
  - 3D bounding boxes
  - 3D center
  - box dimensions
  - yaw orientation
  - object confidence
- CA-1M dataset output:
  - frame-level RGB/RGB-D data
  - pixel-aligned 2D boxes
  - camera-view-consistent 3D boxes
  - world-space object annotations
  - train/validation split

---

## 4. 메모
- 이 논문의 핵심은 **3D point cloud를 모델 입력으로 쓰지 않아도, 충분히 크고 정확한 RGB(-D) frame-level 3D box dataset이 있으면 image-based 3D detector가 강력해질 수 있다**는 주장이다.
- CA-1M은 기존 dataset의 한계였던 작은 객체 부족, noisy annotation source, image reprojection misalignment 문제를 해결하려는 dataset이다.
- CuTR은 3D inductive bias를 줄이고, ViT + DETR-style 2D detector 구조를 거의 그대로 유지하면서 3D box head만 추가한 형태에 가깝다.
- Point-based model은 FARO-derived ground-truth depth를 사용할 때 성능이 크게 좋아진다. 이는 noisy commodity depth에서 backprojection/voxelization 같은 hard operation이 취약할 수 있음을 시사한다.
- CuTR은 RGB-only에서도 의미 있는 성능을 보이며, CA-1M pretraining 후 SUN RGB-D 같은 작은 dataset에서도 point-based method를 능가할 수 있음을 보인다.
- YOLOE + depth completion head 구조와 연결하면, CuTR-style 3D box head를 YOLOE backbone/neck 뒤에 붙이는 방향이 자연스럽다.
- 가능한 결합 구조:
  - YOLOE open-vocabulary 2D detection/segmentation head
  - depth completion head
  - CuTR-style 3D box predictor
  - projected center xy + depth z + dimension + yaw 직접 예측
- 실내 집 환경에서 3D object GT를 만들거나, robot-view RGB-D frame에서 3D cuboid를 예측하는 baseline으로 매우 참고 가치가 높다.
- 다만 CA-1M은 class-agnostic exhaustive object box 중심이며, open-vocabulary semantic label 자체가 핵심은 아니다. YOLOE와 결합할 경우 semantic/open-vocab 부분은 YOLOE가 담당하고, 3D geometry/box regression은 CuTR-style head가 담당하는 구조가 적절하다.

---

## 5. 적용 포인트
 - CA-1M 데이터셋 3d detection 을 위한 학습DB로 활용
