# ViPE: Video Pose Engine for 3D Geometric Perception

- **학회:** NVIDIA Research Whitepapers / arXiv:2508.10934, 2025  
  - 정식 학회 발표 논문이라기보다는 NVIDIA Research whitepaper 및 arXiv technical report 형태로 공개됨.
- **링크:** https://arxiv.org/abs/2508.10934  
  - Project Page: https://research.nvidia.com/labs/toronto-ai/vipe/
- **코드:** https://github.com/nv-tlabs/vipe
- **분야:** 3D Geometric Perception, Visual SLAM, SfM, Camera Pose Estimation, Intrinsics Estimation, Dense Depth Estimation, Spatial AI Dataset Annotation

---

## 1. 요약

- **ViPE**는 보정되지 않은 일반 비디오(raw in-the-wild video)로부터 **camera intrinsics, camera pose, dense near-metric depth map**을 추정하는 video pose estimation engine이다.
- 기존 SLAM은 효율적이고 긴 sequence 처리에 강하지만, known intrinsics, static scene, hand-crafted feature matching에 많이 의존한다.
- 반대로 feed-forward 3D perception 모델은 강인하지만, 긴 비디오 처리 시 GPU memory와 계산량이 커져 확장성이 떨어진다.
- ViPE는 이 둘을 결합하여, **learned optical flow / sparse keypoint / monocular metric depth prior**를 **Bundle Adjustment(BA)** 안에 통합한다.
- 입력 비디오는 calibration target, known camera model, pre-computed pose 없이 처리할 수 있다.
- keyframe 기반 SLAM 구조를 사용하여 arbitrary-length video에 대해 확장성을 확보한다.
- BA 내부에서는 dense flow constraint, sparse point constraint, depth regularization을 함께 최적화한다.
- 후처리 단계에서는 BA에서 얻은 pose-consistent depth와 video depth model의 high-resolution depth를 정렬하여 최종 dense depth를 생성한다.
- pinhole, wide-angle/fisheye, 360° panorama 등 다양한 camera model을 지원한다.
- ViPE는 대규모 비디오 annotation에도 사용되어 Dynpose-100K++, Wild-SDG-1M, Web360 등 약 96M frame 규모의 pose/depth annotation dataset을 생성한다.

---

## 2. 핵심 기여

- **고전적 SLAM과 학습 기반 모델의 결합**
  - DROID-SLAM 계열의 dense BA framework를 기반으로 하되, learned dense optical flow, CUDA 기반 sparse keypoint tracking, monocular metric depth prior를 함께 사용한다.
  - feed-forward 모델처럼 강인한 visual prior를 활용하면서도, 최종 결과는 BA를 통해 multi-frame geometry consistency를 만족하도록 최적화한다.

- **Uncalibrated in-the-wild video에 대한 camera pose / intrinsics / depth 동시 추정**
  - GeoCalib으로 초기 intrinsics를 추정하고, 이후 BA backend에서 camera intrinsics를 함께 refinement한다.
  - camera pose, low-resolution keyframe depth, intrinsics를 joint optimization으로 추정한다.
  - metric depth prior를 사용해 monocular SLAM의 scale ambiguity 및 scale drift를 완화한다.

- **대규모 annotation에 적합한 dense depth alignment와 dataset release**
  - BA depth는 pose와 일관성이 좋지만 low-resolution/noisy하고, video depth는 high-resolution/temporally smooth하지만 absolute scale이 불안정하다.
  - ViPE는 두 depth를 inverse-depth affine alignment로 결합해 pose-consistent high-resolution dense metric depth를 생성한다.
  - 이 pipeline으로 real-world internet video, AI-generated video, panoramic video를 포함하는 대규모 annotated dataset을 공개한다.

---

## 3. 방법

### 입력

- **입력 비디오:** 보정되지 않은 monocular RGB video
- **사전 정보:**
  - camera calibration 불필요
  - pose prior 불필요
  - depth sensor 불필요
- **선택적 입력/설정:**
  - dynamic object class list
  - camera model type: pinhole, wide-angle/fisheye, panorama 등
- **내부적으로 사용하는 모델/모듈:**
  - GeoCalib: initial intrinsics estimation
  - DROID-SLAM style dense flow network: dense optical flow constraint
  - cuVSLAM: Shi-Tomasi + Lucas-Kanade 기반 sparse keypoint tracking
  - Metric3Dv2 / UniDepthV2 / UniK3D: monocular metric depth prior
  - Video depth estimation model: temporally smooth high-resolution depth
  - GroundingDINO + SAM + XMem: dynamic object mask propagation
  - PriorDA: sparse/incomplete BA depth infilling

### 핵심 아이디어

#### 1. Keyframe 기반 SLAM 구조

ViPE는 전체 비디오를 모든 frame 기준으로 직접 최적화하지 않고, keyframe을 중심으로 graph를 구성한다.

1. 비디오에서 4개 frame을 균일 샘플링한다.
2. GeoCalib을 사용해 초기 camera intrinsics를 얻는다.
3. 각 incoming frame과 이전 keyframe 사이의 motion을 추정한다.
4. motion이 threshold보다 크면 keyframe으로 추가한다.
5. 최근 keyframe들에 대해 sliding-window frontend BA를 수행한다.
6. keyframe 수가 8, 16, 64개에 도달했을 때와 tracking 종료 시점에 backend full BA를 수행한다.
7. non-keyframe은 가장 가까운 두 keyframe에 연결해 pose infilling을 수행한다.

#### 2. BA formulation

ViPE는 각 keyframe에 대해 다음 unknown을 최적화한다.

- camera pose: \(T_i \in SE(3)\)
- camera intrinsics: \(k\)
- low-resolution depth map: \(D_i \in \mathbb{R}^{h \times w}\)

전체 energy는 다음 세 항으로 구성된다.

\[
e_{ViPE}
=
\sum_{(i,j) \in \mathcal{E}} e_{dense}
+
\sum_{(i,j) \in \mathcal{E}} e_{sparse}
+
\alpha \sum_{i \in \mathcal{V}} e_{depth}
\]

각 항의 의미는 다음과 같다.

- \(e_{dense}\): dense optical flow 기반 reprojection consistency
- \(e_{sparse}\): sparse keypoint track 기반 고해상도 localization constraint
- \(e_{depth}\): monocular metric depth prior 기반 depth regularization

최적화는 Gauss-Newton solver로 수행되며, sparse linear system은 COLAMD reordering을 사용해 효율적으로 푼다.

#### 3. Dense flow constraint

현재 pose와 depth로 frame \(i\)의 pixel을 frame \(j\)로 projection했을 때의 이동량과, flow network가 예측한 optical flow가 일치하도록 한다.

핵심 residual은 다음 형태이다.

\[
\Pi_k(T_j^{-1}T_i \circ \Pi_k^{-1}(D_i[u])) - u - F_{ij}[u]
\]

이 항은 textureless region에서도 learned flow prior를 이용해 dense correspondence를 제공한다.

#### 4. Sparse point constraint

Dense flow는 low-resolution에서 동작하므로, 고해상도 이미지의 corner-like feature를 놓칠 수 있다. 이를 보완하기 위해 cuVSLAM 기반 sparse keypoint track을 추가한다.

- Shi-Tomasi corner detector로 feature를 검출한다.
- Lucas-Kanade tracker로 frame 간 feature를 추적한다.
- sparse feature는 원본 high-resolution image에서 계산되므로 sub-pixel 수준의 localization 정보를 제공한다.

초기 formulation은 sparse keypoint 위치에서 depth를 bilinear interpolation하는 방식이지만, Hessian 구조가 복잡해진다. 따라서 논문에서는 bilinear splatting을 사용해 sparse flow를 low-resolution grid에 누적하고, dense flow term과 유사한 형태로 변환한다.

#### 5. Depth regularization

작은 camera motion이나 degenerate motion에서는 pose-depth ambiguity가 커진다. 이를 줄이기 위해 pretrained monocular metric depth estimator가 제공하는 depth를 prior로 사용한다.

\[
e_{depth}(D_i)
=
\sum_u m[u] \cdot \|D_i[u] - D_i^{prior}[u]\|^2
\]

이 항은 다음 역할을 한다.

- monocular SLAM의 scale drift 완화
- real-world metric scale 추정 보조
- pose estimation이 불안정한 구간에서 depth를 regularize

Intrinsics가 BA에서 업데이트되면, metric depth model의 prediction도 함께 업데이트한다.

#### 6. Dynamic object masking

실제 비디오에는 사람, 차량, 동물 등 동적 객체가 포함될 수 있다. ViPE는 동적 객체를 pose estimation에서 제외하기 위해 semantic masking을 사용한다.

절차는 다음과 같다.

1. 사용자가 dynamic class list를 지정한다.
2. GroundingDINO가 해당 class의 bounding box를 예측한다.
3. SAM이 segmentation mask를 생성한다.
4. 계산량을 줄이기 위해 일정 frame interval마다만 segmentation을 수행한다.
5. XMem으로 mask를 frame sequence 전체에 propagate한다.
6. dynamic mask를 반전해 static background mask \(M\)을 만든다.
7. dense flow term의 weight map에 \(M\)을 곱하고, sparse point track 중 dynamic region에 있는 track은 제거한다.

#### 7. 다양한 camera model 지원

ViPE는 radial camera formulation을 사용하여 다양한 camera model을 지원한다.

- pinhole camera: \(q_k(\theta)=\tan\theta\)
- wide-angle/fisheye camera: unified camera model 사용
- 360° panorama: panorama를 6개 pinhole camera view(front/back/left/right/up/down)로 projection하여 처리

Multi-camera rig의 경우 rig-to-camera transform \(T_v\)를 포함하도록 BA formulation을 확장한다.

#### 8. Post-processed dense depth alignment

BA에서 얻은 depth는 pose와 잘 맞지만 low-resolution이고 noisy할 수 있다. 반면 video depth model의 depth는 high-resolution이고 temporally smooth하지만 absolute scale이 불안정할 수 있다.

ViPE는 두 depth를 결합한다.

1. video depth model로 high-resolution affine-invariant depth \(D_i^{VDA}\)를 얻는다.
2. BA depth를 unprojection하여 point cloud로 만들고, 여러 keyframe의 point cloud를 aggregate한다.
3. pose consistency check를 통과한 point만 각 frame에 다시 project하여 sparse BA depth \(D_i^{BA}\)를 만든다.
4. inverse-depth 공간에서 affine alignment를 수행한다.

\[
\frac{\alpha_i}{D_i^{VDA}[u]} + \beta_i
\approx
\frac{1}{D_i^{BA}[u]}
\]

5. frame 간 scale/shift flickering을 줄이기 위해 momentum update를 적용한다.

\[
\hat{\alpha}_i = m\hat{\alpha}_{i-1} + (1-m)\alpha_i
\]

\[
\hat{\beta}_i = m\hat{\beta}_{i-1} + (1-m)\beta_i
\]

6. 최종 dense depth는 다음과 같이 계산된다.

\[
D_i^{HD}
=
\frac{1}{\hat{\alpha}_i / D_i^{VDA} + \hat{\beta}_i}
\]

만약 projected BA depth coverage가 부족하면 PriorDA로 infill하고, 극단적으로 거의 coverage가 없으면 monocular metric depth estimator 결과를 fallback으로 사용한다.

### 출력

- frame별 camera pose
- 최적화된 camera intrinsics
- keyframe 기반 SLAM map / point cloud
- frame별 high-resolution dense near-metric depth map
- optional COLMAP-format output
- 대규모 video annotation dataset 생성에 사용할 수 있는 pose/depth/geometric buffer

---

## 4. 메모

- ViPE는 단순히 monocular depth를 예측하는 모델이 아니라, **raw video를 3D geometric annotation으로 변환하는 end-to-end processing engine**에 가깝다.
- 핵심은 learned model이 최종 답을 직접 내는 것이 아니라, learned flow/depth/keypoint prior를 BA의 measurement 또는 regularization으로 사용한다는 점이다.
- Dense flow는 textureless region에서 유리하고, sparse keypoint는 high-resolution localization에 유리하다. ViPE는 두 constraint를 함께 사용해 상호 보완한다.
- Metric depth prior는 monocular SLAM에서 치명적인 scale ambiguity를 줄이는 데 중요하다.
- Post-processed dense depth alignment는 실제 depth 품질 관점에서 매우 중요하다. BA depth의 pose consistency와 video depth의 high-resolution detail을 결합하기 때문이다.
- Kinect depth + LiDAR sparse depth 융합에도 유사한 아이디어를 적용할 수 있다. Kinect depth를 dense prior로 두고 LiDAR projected sparse depth를 metric anchor로 사용해 inverse-depth affine alignment 및 residual correction을 수행하면 된다.
- 단, ViPE의 depth alignment는 완전한 sensor fusion이라기보다 **pose-consistent sparse/dense depth anchor를 이용한 video depth scale/shift 보정**에 가깝다.
- 실제 적용 시 dynamic object masking, time synchronization, calibration accuracy, occlusion filtering이 결과 품질에 큰 영향을 준다.
- 평가에서는 TUM RGB-D, KITTI/RDS, OpenDV, VidBench, Sintel, ETH3D 등 다양한 benchmark를 사용하며, pose/intrinsics/depth 성능을 비교한다.
- 대규모 dataset release 측면에서 ViPE는 spatial AI, world model, video generation, robotics policy learning을 위한 geometry annotation pipeline으로 볼 수 있다.

---

## 5. 적용 포인트
 - XMem 알고리즘을 통한 frame간 segment tracking 방법
 - 두 depth를 inverse depth에서 align(least square)하는 방법
 - public dataset(depth, pose, intrinsic)를 포함하는 dataset 활용 가능
