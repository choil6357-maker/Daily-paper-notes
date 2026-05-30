# TUN3D: Towards Real-World Scene Understanding from Unposed Images

- **학회:** ICRA 2026
- **링크:** https://arxiv.org/abs/2509.21388
- **코드:** https://github.com/col14m/TUN3D
- **분야:** Indoor Scene Understanding, 3D Object Detection, Layout Estimation, Multi-view 3D Reconstruction, Unposed Image-based 3D Understanding

---

## 1. 요약

- **TUN3D**는 실내 장면에서 **3D 객체 검출**과 **룸 레이아웃 추정**을 하나의 모델로 동시에 수행하는 방법이다.
- 기존 실내 3D scene understanding 방법들은 주로 **point cloud 입력**에 의존했지만, TUN3D는 이를 **multi-view image 입력**까지 확장한다.
- 특히 TUN3D는 **GT camera pose나 depth supervision 없이도** unposed multi-view images에서 동작하는 것을 목표로 한다.
- 전체 파이프라인은 입력 modality에 따라 다르지만, 최종적으로는 point cloud 형태로 변환한 뒤 동일한 TUN3D network를 적용한다.
- Point cloud 입력에서는 colored point cloud를 2 cm voxel로 변환한 뒤 sparse 3D CNN backbone으로 처리한다.
- Posed image 입력에서는 DUSt3R가 dense depth를 추정하고, GT camera pose와 함께 TSDF fusion을 수행해 point cloud를 생성한다.
- Unposed image 입력에서는 DUSt3R가 depth map과 camera parameter를 모두 추정하고, 추정 pose 기반 TSDF fusion으로 point cloud를 만든다.
- 네트워크는 TR3D 계열의 lightweight sparse-convolutional backbone/neck을 기반으로 하며, detection head와 layout head를 분리해 사용한다.
- 핵심 기여 중 하나는 벽을 **2×2D offsets + height**로 표현하는 BEV 기반 wall parameterization이다.
- 실험 결과, TUN3D는 GT point cloud, posed images, unposed images 세 가지 설정 모두에서 joint layout estimation과 3D object detection의 새로운 SOTA를 달성한다.

---

## 2. 핵심 기여

- **입력 modality 요구사항 완화**
  - 기존 point cloud 기반 실내 scene understanding을 multi-view images로 확장한다.
  - camera pose가 있는 이미지뿐 아니라, pose가 없는 일반 이미지/비디오에서도 동작할 수 있도록 설계했다.
  - depth sensor나 tracker가 없는 일반 소비자용 카메라 환경에서도 적용 가능성을 보인다.

- **Joint layout estimation + 3D object detection 구조 제안**
  - 하나의 sparse 3D CNN 기반 backbone에서 scene feature를 공유하고, 두 개의 task-specific head로 object와 layout을 동시에 예측한다.
  - 3D object detection은 TR3D와 유사한 anchor-free sparse voxel location 기반 예측 구조를 사용한다.
  - Layout estimation은 wall classification과 wall geometry regression으로 구성된다.

- **효율적인 BEV 기반 wall parameterization 제안**
  - 기존 PQ-Transformer의 wall center/length/height/normal 방식이나 4-corner 3D offset 방식보다 더 compact하고 안정적인 표현을 제안한다.
  - 벽을 BEV 평면의 두 lower corner offset과 wall height만으로 표현한다.
  - 이 방식은 5개 parameter만 사용하면서도 malformed wall geometry를 줄이고 layout F1을 향상시킨다.

- **세 가지 입력 설정에서 SOTA 달성**
  - GT point cloud 입력
  - posed multi-view images 입력
  - unposed multi-view images 입력
  - ScanNet, S3DIS, ARKitScenes, Structured3D 등 다양한 benchmark에서 기존 방법 대비 우수한 성능을 보인다.

---

## 3. 방법

### 입력

TUN3D는 세 가지 입력 시나리오를 다룬다.

#### 1) Colored point cloud

```text
P = {p_i}_{i=1}^{N}, p_i = (x_i, y_i, z_i, r_i, g_i, b_i)
```

- 각 point는 3D 좌표와 RGB 색상을 가진다.
- 입력 point cloud는 2 cm voxel size로 voxelization된다.
- 이후 sparse 3D CNN backbone에 입력된다.

#### 2) Posed multi-view images

```text
{I_m}_{m=1}^{M}, K_m, T_m
```

- 입력은 여러 장의 RGB image, camera intrinsic, camera extrinsic이다.
- DUSt3R를 pose-aware mode로 사용해 dense depth map을 추정한다.
- 추정 depth와 GT camera pose를 이용해 TSDF fusion을 수행한다.
- TSDF volume에서 point cloud를 추출한 뒤 TUN3D point cloud pipeline에 입력한다.

#### 3) Unposed multi-view images

```text
{I_m}_{m=1}^{M}
```

- 입력은 camera pose, intrinsic, depth가 없는 image collection이다.
- DUSt3R가 dense depth map과 camera parameter를 함께 추정한다.
- 추정 pose와 depth를 이용해 TSDF fusion을 수행한다.
- 생성된 point cloud를 TUN3D에 입력한다.

---

### 핵심 아이디어

#### 1) 입력을 point cloud representation으로 통일

TUN3D의 핵심 처리기는 point cloud 기반이다. 따라서 image 입력의 경우에도 바로 image feature로 object/layout을 예측하지 않고, 먼저 DUSt3R와 TSDF fusion을 통해 point cloud로 변환한다.

```text
GT point cloud
        ↓
TUN3D

posed images
        ↓
DUSt3R depth estimation + GT pose TSDF fusion
        ↓
point cloud
        ↓
TUN3D

unposed images
        ↓
DUSt3R depth/pose estimation + estimated pose TSDF fusion
        ↓
point cloud
        ↓
TUN3D
```

이 구조 덕분에 point cloud, posed images, unposed images를 하나의 downstream scene understanding model로 처리할 수 있다.

#### 2) Sparse 3D CNN backbone + neck

- Backbone은 TR3D/FCAF3D 계열의 sparse 3D ResNet 구조를 따른다.
- 입력 point cloud는 2 cm voxel로 변환된다.
- Sparse convolution residual block을 거치며 8 cm, 16 cm, 32 cm, 64 cm feature level을 생성한다.
- 최대 channel 수는 효율성을 위해 128로 제한한다.
- Neck은 여러 resolution의 voxel feature를 통합한다.
- 32 cm와 64 cm level에서는 sparse generative convolution을 사용해 visibility field가 지나치게 줄어드는 문제를 완화한다.

#### 3) Detection head

Detection head는 TR3D와 유사한 구조이다.

- 16 cm와 32 cm feature level에서 object prediction을 수행한다.
- 각 sparse voxel location \(\hat{v}_j\)마다 다음을 예측한다.
  - class logits
  - object center offset
  - 3D box log-size

예측 box는 다음과 같이 계산된다.

```text
center = voxel location + predicted offset
size   = exp(predicted log-size)
```

학습 시 각 GT object는 해당 object category가 담당하는 feature level에서 object center와 가장 가까운 6개 sparse location에 할당된다. 이 6개 location이 해당 object를 예측하는 positive sample이 된다.

#### 4) Layout head

Layout head는 32 cm level feature를 사용해 wall layout을 예측한다.

각 candidate location에 대해 다음을 예측한다.

- wall / non-wall classification score
- wall geometry parameter

TUN3D의 핵심은 wall geometry를 3D corner 전체가 아니라 BEV 기반으로 간결하게 표현하는 것이다.

#### 5) BEV 기반 wall parameterization

기존 방식들은 다음과 같은 한계가 있다.

- PQ-Transformer 방식: wall center, length, height, normal을 따로 예측하므로 geometry가 불안정할 수 있다.
- 4×3D offsets 방식: 네 corner를 모두 직접 예측하므로 자유도가 너무 높고 malformed wall이 생길 수 있다.
- 2×3D offsets + height 방식: 더 간결하지만 여전히 3D offset 간 제약이 약하다.

TUN3D는 벽이 수직 방향으로 쌓이지 않는다는 indoor layout의 구조적 가정을 이용해, wall prediction을 BEV plane에서 수행한다.

하나의 wall은 다음 5개 parameter로 표현된다.

```text
wall = lower corner 1의 2D offset
     + lower corner 2의 2D offset
     + wall height
```

수식적으로는 다음과 같다.

```text
Δu_j^(1), Δu_j^(2) ∈ R^2
h_j ∈ R_+
```

Lower corner는 BEV 평면에서 계산하고, upper corner는 height 방향으로 올린다.

```text
q_j^(L,m) = (u_hat_j + Δu_j^(m), 0)
q_j^(U,m) = q_j^(L,m) + h_j e_z
```

이 방식은 parameter 수가 적고, wall geometry가 더 rigid해지며, layout prediction 품질을 향상시킨다.

#### 6) Height distribution encoding

BEV projection을 하면 z축 height 정보가 손실된다. 이를 보완하기 위해 TUN3D는 scene-level height feature를 추가한다.

- scene 내 point들의 z-coordinate을 이용해 10개의 z-quantile을 계산한다.
- 10개 z-quantile을 3-layer MLP와 ReLU로 encoding한다.
- size 40 vector로 변환한다.
- 이를 128-channel floor-projected feature에 concatenate한다.
- 최종 floor-projected representation은 168 channel이 된다.

```text
128-channel BEV feature + 40-dimensional height vector = 168-channel feature
```

#### 7) Training assignment와 loss

Object와 wall은 서로 다른 assignment rule을 사용한다.

- Object assignment
  - object category별로 담당 feature level을 미리 정한다.
  - 큰 object는 주로 32 cm level, 작은 object는 16 cm level에서 처리한다.
  - 각 GT object는 중심에 가장 가까운 6개 location에 할당된다.

- Wall assignment
  - wall은 large object처럼 취급한다.
  - 각 GT wall은 32 cm level에서 가장 가까운 6개 location에 할당된다.
  - BEV parameterization을 사용할 경우 2D floor projection 위치를 기준으로 matching한다.

전체 loss는 다음 네 항의 합이다.

```text
L = L_det_focal + L_det_DIoU + L_layout_focal + L_layout_L1
```

- \(L^{det}_{focal}\): object classification
- \(L^{det}_{DIoU}\): 3D bounding box regression
- \(L^{layout}_{focal}\): wall classification
- \(L^{layout}_{L1}\): wall parameter regression

---

### 출력

TUN3D의 출력은 compact하고 semantic한 indoor scene representation이다.

#### 1) 3D object detection 결과

```text
O = {(b_k, c_k)}_{k=1}^{K}
```

각 object는 다음을 포함한다.

- object category
- 3D bounding box center
- 3D bounding box size

```text
b_k = (t_k, s_k)
```

#### 2) Layout estimation 결과

```text
W = {w_l}_{l=1}^{L}
```

각 wall은 네 corner의 3D 좌표로 표현된다.

```text
w_l = (q_l,1, q_l,2, q_l,3, q_l,4)
```

TUN3D 내부에서는 wall을 BEV lower corners + height로 예측하지만, 최종 출력은 3D wall corners로 복원된다.

#### 3) 최종 scene representation

```text
scene = 3D object boxes + object categories + wall layout
```

이는 dense mesh나 dense point cloud보다 훨씬 가볍지만, 실내 공간의 구조와 주요 object semantics를 함께 담는 표현이다.

---

## 4. 메모

- TUN3D는 이름 그대로 **unposed images**까지 처리하는 것을 강조하지만, 실제 core network는 point cloud 기반 sparse 3D CNN이다.
- Image-only setting에서는 DUSt3R의 reconstruction 품질이 downstream 성능에 직접적인 영향을 준다.
- 논문에서는 DUSt3R가 DROID-SLAM보다 더 정밀한 geometry를 생성해 layout estimation과 3D object detection 성능을 크게 향상시킨다고 보고한다.
- 최종 TUN3D는 45 frames를 사용한다. 이는 ImVoxelNet의 50 images, NeRF-Det의 100 images와 비교해 비슷하거나 더 적은 수준이다.
- TUN3D는 LLM 기반 SpatialLM보다 훨씬 빠르고, PQ-Transformer보다 약 4배 빠르다고 보고한다.
- UniDet3D 기반 layout 확장과 비교하면, TUN3D는 1.7배 빠르면서 layout F1도 더 높다.
- 10개의 z-quantile을 사용하는 height distribution encoding은 계산 overhead가 거의 없으면서 ScanNet과 S3DIS에서 layout F1을 의미 있게 향상시킨다.
- 제안 wall parameterization은 단 5개 parameter만 사용하면서 ScanNet layout F1을 향상시킨다.
- 실내 주거 환경에서 3D bbox annotation guide나 compact scene parsing 용도로는 유용할 가능성이 높다.
- 다만 dense reconstruction 자체를 목표로 하는 방법은 아니므로, depth completion GT나 mesh-level reconstruction 품질을 직접 보장하지는 않는다.

---

## 5. 적용 포인트
 - wall을 표현하는 방법
 - 3d parametric reconstruction baseline으로 활용 가능
