# Gaussian Splatting SLAM

- **학회:** CVPR 2024 Highlight / Best Demo Award
- **링크:** https://arxiv.org/abs/2312.06741
- **코드:** https://github.com/muskie82/MonoGS
- **분야:** Visual SLAM, Dense SLAM, 3D Gaussian Splatting, Monocular SLAM, RGB-D SLAM, Novel View Synthesis

---

## 1. 요약
- 본 논문은 **3D Gaussian Splatting(3DGS)을 SLAM의 유일한 3D map representation으로 사용하는 최초의 online visual SLAM 시스템**을 제안한다.
- 기존 3DGS는 보통 SfM/COLMAP 등으로 미리 구한 정확한 camera pose를 필요로 했지만, 이 논문은 SLAM 환경에서 **camera pose와 3D Gaussian map을 online으로 함께 최적화**한다.
- 입력은 **monocular RGB**를 기본으로 하며, depth sensor가 있는 경우 **RGB-D SLAM**으로 자연스럽게 확장된다.
- 시스템은 3D Gaussian map을 rendering한 RGB/depth와 실제 관측 RGB/depth 간의 residual을 줄이는 방식으로 tracking과 mapping을 수행한다.
- 3DGS의 빠른 differentiable rasterisation을 활용하여 near real-time으로 동작하며, 논문에서는 약 **3 FPS** 수준의 live SLAM을 보고한다.
- Monocular 환경에서 depth ambiguity와 잘못된 Gaussian geometry 문제를 줄이기 위해 **isotropic regularisation**, **geometric verification**, **Gaussian pruning**, **keyframe covisibility management**를 도입한다.
- 평가에서는 TUM RGB-D, Replica dataset 등을 사용하며, monocular 및 RGB-D 환경 모두에서 trajectory estimation과 rendering quality 측면의 경쟁력 있는 성능을 보인다.
- 단, Gaussian은 명시적인 surface를 표현하지 않으므로, surface normal, mesh, 정확한 point-cloud GT를 얻으려면 별도의 geometry extraction 또는 rendered depth 기반 후처리가 필요하다.

---

## 2. 핵심 기여
- **3DGS-only SLAM**
  - Sparse feature map, voxel grid, TSDF, mesh, neural field 등을 별도로 섞지 않고, 3D Gaussian만으로 tracking, mapping, rendering을 통합한다.
  - Gaussian의 mean, covariance/scale/rotation, color, opacity가 map의 핵심 parameter가 된다.

- **Direct camera pose optimisation**
  - 현재 camera pose에서 Gaussian map을 rendering한 결과와 입력 image를 직접 비교하여 pose를 최적화한다.
  - 이를 위해 SE(3) Lie group 상에서 camera pose에 대한 **analytic Jacobian**을 유도하여 빠른 tracking을 가능하게 한다.

- **Online dense reconstruction 안정화**
  - Incremental SLAM에서 발생하는 잘못된 Gaussian, elongated Gaussian, floating artifact를 줄이기 위해 다음 기법을 사용한다.
    - Gaussian covisibility 기반 keyframe selection
    - Gaussian insertion and pruning
    - Isotropic Gaussian shape regularisation
    - RGB-D 입력 시 photometric residual + geometric residual 동시 최적화
    - Occlusion-aware visibility estimation

---

## 3. 방법

### 입력
- **Monocular mode**
  - RGB image sequence
  - Camera intrinsic
  - Depth sensor는 사용하지 않음
  - Pre-trained monocular depth predictor도 사용하지 않음

- **RGB-D mode**
  - RGB image sequence
  - Aligned depth image sequence
  - Camera intrinsic
  - Depth는 Gaussian 초기화와 geometric residual에 사용됨

---

### 핵심 아이디어
- 장면을 3D Gaussian들의 집합으로 표현한다.

\[
G_i = \{\mu_i, \Sigma_i, c_i, \alpha_i\}
\]

  - \(\mu_i\): Gaussian mean, 즉 world 좌표계의 3D 중심 위치
  - \(\Sigma_i\): Gaussian의 shape, scale, orientation
  - \(c_i\): color
  - \(\alpha_i\): opacity

- **Tracking**
  - 새 frame이 들어오면 기존 Gaussian map은 고정하고, 현재 camera pose만 최적화한다.
  - Gaussian map을 현재 pose에서 rendering한 RGB image와 실제 입력 RGB image의 차이를 줄인다.

\[
E_{pho} = \|I(G, T_{CW}) - \bar{I}\|_1
\]

  - RGB-D mode에서는 rendered depth와 observed depth도 비교한다.

\[
E_{geo} = \|D(G, T_{CW}) - \bar{D}\|_1
\]

  - 즉, tracking은 “현재 camera pose를 기존 Gaussian map에 맞추는 과정”이다.

- **Keyframing**
  - 모든 frame을 map optimisation에 사용하면 계산량이 너무 크므로, covisibility와 relative translation을 기준으로 keyframe을 선택한다.
  - Gaussian들이 ray 방향으로 정렬되어 rendering되므로, occlusion-aware visibility를 이용해 keyframe 간 covisibility를 계산할 수 있다.

- **Gaussian insertion**
  - 새 keyframe이 선택되면 새롭게 보이는 영역에 Gaussian을 추가한다.
  - RGB-D mode에서는 depth를 back-projection하여 Gaussian mean을 초기화한다.

\[
p^C = d K^{-1}[u,v,1]^T
\]

\[
p^W = T_{WC}p^C
\]

  - Monocular mode에서는 기존 Gaussian map에서 rendering한 depth 또는 median depth를 이용해 Gaussian을 초기화한다.

- **Mapping**
  - 현재 local keyframe window \(W_k\)와 일부 random past keyframes \(W_r\)를 사용하여 Gaussian map을 최적화한다.

\[
W = W_k \cup W_r
\]

  - 각 keyframe에서 Gaussian map을 rendering했을 때 실제 RGB/depth 관측과 잘 맞도록 Gaussian parameter와 일부 keyframe pose를 업데이트한다.

\[
\min_{T^k_{CW}, G} \sum_{k \in W} E^k_{pho} + \lambda_{iso}E_{iso}
\]

  - RGB-D mode에서는 \(E_{geo}\)도 추가된다.

- **Isotropic regularisation**
  - 3DGS는 ray 방향으로 Gaussian이 길게 늘어나는 artifact가 생길 수 있다.
  - 이를 줄이기 위해 Gaussian scale이 특정 방향으로 과도하게 커지는 것을 penalise한다.

\[
E_{iso} = \sum_i \|s_i - \tilde{s}_i \cdot \mathbf{1}\|_1
\]

  - 목적은 Gaussian이 지나치게 elongated되지 않게 하고, geometry consistency를 높이는 것이다.

---

### 출력
- **3D Gaussian map**
  - 각 Gaussian의 mean, covariance/scale/rotation, color, opacity로 구성된 map
  - Gaussian mean은 point cloud의 3D 좌표처럼 해석할 수 있지만, 반드시 실제 surface point라고 보장되지는 않는다.

- **Estimated camera trajectory**
  - 각 keyframe 또는 frame의 camera pose
  - Rendered depth나 입력 depth를 global point cloud로 변환할 때 핵심적으로 사용된다.

- **Rendered RGB image**
  - Gaussian map을 특정 camera pose에서 rendering한 novel view image

- **Rendered depth map**
  - Gaussian mean의 camera-frame depth를 alpha blending하여 얻은 view-dependent depth map
  - 필요하면 intrinsic과 estimated pose를 이용해 back-projection하여 point cloud로 변환할 수 있다.

---

## 4. 메모
- RGB-D case에서 depth는 단순히 Gaussian 초기화에만 사용되는 것이 아니라, tracking/mapping 중 geometric residual로도 사용된다.
- Rendered RGB 과정에도 Gaussian의 depth 정보가 projection, visibility ordering, alpha blending에 사용된다. 하지만 RGB loss만으로는 geometry가 충분히 제약되지 않기 때문에 RGB-D에서는 depth residual이 중요하다.
- Rendered depth는 입력 depth와 유사해지도록 최적화되지만, rendered depth 값이 Gaussian mean에 그대로 대입되는 것은 아니다. Depth residual의 gradient가 Gaussian mean, shape, opacity, pose를 업데이트한다.
- 3DGS-SLAM에는 TSDF/mesh/point cloud처럼 명시적인 surface map이 저장되지는 않는다. 대신 Gaussian mean과 opacity/scale/rotation이 geometry를 암묵적으로 표현한다.
- Rendered depth를 back-projection해 point cloud를 만들려면 estimated camera pose가 반드시 필요하다.
- 따라서 3DGS-SLAM 결과를 3D object detection GT, 3D bbox, mesh, dense depth GT 생성에 활용하려면 **camera pose accuracy**와 **global consistency** 검증이 매우 중요하다.
- 논문 결론에서도 large-scale scene을 위한 loop closure 통합과 surface normal 등 explicit geometry extraction을 향후 연구 방향으로 언급한다.

---

## 5. 적용 포인트
 - 
