# Depth Anything with Any Prior (Prior Depth Anything)

- **학회:** ICLR 2026 Poster
- **링크:** https://arxiv.org/abs/2505.10565 / https://prior-depth-anything.github.io/
- **코드:** https://github.com/SpatialVision/Prior-Depth-Anything
- **분야:** Monocular Depth Estimation, Prior-based Metric Depth Estimation, Depth Completion, Depth Super-resolution, Depth Inpainting, 3D Reconstruction

---

## 1. 요약

- **Prior Depth Anything**은 불완전하지만 metric scale이 정확한 depth measurement와, dense하고 기하 구조가 좋은 monocular depth prediction을 결합하여 **dense metric depth map**을 생성하는 프레임워크이다.
- 기존 MDE 모델은 이미지 전반의 relative geometry와 fine detail을 잘 예측하지만, absolute metric scale은 부족하다.
- 반대로 depth sensor, SfM, LiDAR, low-resolution depth 등은 metric 정보를 제공하지만 sparse, low-res, hole, noise 같은 불완전성을 가진다.
- 본 논문은 이 두 정보를 **coarse-to-fine pipeline**으로 결합한다.
- 첫 번째 단계에서는 frozen MDE 모델의 relative depth prediction을 이용해 불완전한 metric prior를 채우는 **coarse metric alignment**를 수행한다.
- 이때 단순 interpolation이나 global alignment 대신, 각 missing pixel 주변의 valid metric depth를 이용해 local scale/shift를 추정하는 **pixel-level metric alignment**를 사용한다.
- 추가로 query pixel과 가까운 supporting point에 더 큰 weight를 주는 **distance-aware re-weighting**을 적용하여 더 부드럽고 안정적인 pre-filled prior를 만든다.
- 두 번째 단계에서는 RGB image, pre-filled metric prior, relative depth prediction을 condition으로 사용하는 **conditioned MDE model**을 통해 noise와 misalignment를 보정한다.
- 하나의 모델로 zero-shot depth completion, depth super-resolution, depth inpainting을 모두 처리할 수 있으며, mixed prior 상황에서도 기존 task-specific 방법보다 강건하다.
- VGGT 같은 3D reconstruction foundation model의 depth prediction을 refine하는 plug-and-play module로도 활용 가능하다.

---

## 2. 핵심 기여

- **Any prior를 처리하는 통합 metric depth framework 제안**
  - Sparse points, LiDAR-like prior, extremely sparse prior, low-resolution depth, range/shape/object missing area 등 다양한 prior pattern을 하나의 구조에서 처리한다.
  - Completion, super-resolution, inpainting을 별도 모델이 아니라 하나의 모델로 수행한다.

- **Coarse Metric Alignment 제안**
  - Frozen MDE 모델이 만든 dense relative depth prediction을 이용해 incomplete metric prior를 dense하게 pre-fill한다.
  - 각 missing pixel마다 주변 valid prior point를 kNN으로 찾고, local scale/shift를 least squares로 추정해 prediction을 metric scale로 변환한다.
  - Distance-aware weighting을 통해 가까운 valid measurement를 더 신뢰하도록 하여 discontinuity와 alignment error를 줄인다.

- **Fine Structure Refinement 설계**
  - Coarse alignment 결과는 metric scale은 좋지만 prior noise에 민감할 수 있다.
  - 이를 해결하기 위해 RGB image, pre-filled prior, relative prediction을 condition으로 받는 conditioned MDE model을 학습한다.
  - Metric condition은 absolute scale을 제공하고, geometry condition은 boundary/detail/relative structure를 제공한다.

- **Zero-shot 성능 및 mixed prior robustness 입증**
  - NYUv2, ScanNet, ETH3D, DIODE, KITTI, ARKitScenes, RGB-D-D 등 7개 real-world dataset에서 평가한다.
  - Depth completion, super-resolution, inpainting 각각의 task-specific baseline과 비교해 동등하거나 더 우수한 성능을 보인다.
  - 서로 다른 prior pattern이 섞인 mixed prior 상황에서도 성능 저하가 작다.

- **Test-time model switching 가능**
  - Frozen MDE model을 test-time에 교체할 수 있어 accuracy-efficiency trade-off를 조절할 수 있다.
  - 더 강한 MDE 모델을 사용하면 성능 향상을 얻고, 작은 모델을 사용하면 latency를 줄일 수 있다.

---

## 3. 방법

### 입력

- **RGB image**
  - `I ∈ R^{3×H×W}`

- **Metric depth prior**
  - `D_prior ∈ R^{H×W}`
  - 형태는 제한하지 않는다.
  - 예:
    - SfM sparse points
    - LiDAR sparse points
    - extremely sparse points
    - captured low-resolution depth
    - ×8 / ×16 downsampled depth
    - range missing depth
    - square/shape missing region
    - object mask missing region
    - mixed prior

- **Frozen MDE prediction**
  - RGB image를 frozen MDE model에 넣어 얻은 dense relative depth
  - `D_pred ∈ R^{H×W}`

---

### 핵심 아이디어

#### 1) Frozen MDE로 dense relative geometry 생성

먼저 RGB image를 frozen MDE model에 입력하여 dense relative depth prediction을 얻는다.

```text
D_pred = FrozenMDE(I)
```

`D_pred`는 dense하고 fine geometry는 좋지만 metric scale은 정확하지 않다.

---

#### 2) Coarse Metric Alignment로 incomplete prior pre-fill

측정 prior에서 valid depth가 있는 pixel set을 `P`라고 둔다.

```text
P = {(x_i, y_i)} where D_prior(x_i, y_i) is valid
```

Valid pixel은 측정값을 유지한다.

```text
D_hat_prior(x, y) = D_prior(x, y),  if (x, y) ∈ P
```

원문에는 valid pixel에서 `D_hat_prior = D_pred`처럼 보이는 식이 있으나, 설명상 valid metric prior를 상속하는 것이 자연스럽다. 따라서 구현 관점에서는 `D_prior`를 유지하는 방식이 타당하다.

Missing pixel `q = (x_hat, y_hat)`에 대해서는 다음 과정을 수행한다.

1. `P`에서 q와 가장 가까운 K개 valid point를 kNN으로 찾는다.
   - 논문 기본값: `K = 5`

2. 주변 K개 point에서 local scale `s`와 shift `t`를 추정한다.

```text
s, t = argmin_{s,t} Σ_k w_k || s * D_pred(x_k, y_k) + t - D_prior(x_k, y_k) ||^2
```

3. Distance-aware weight를 적용한다.

```text
w_k = 1 / ( || q - p_k ||^2 + ε )
```

4. Missing pixel을 metric scale로 변환된 prediction으로 채운다.

```text
D_hat_prior(q) = s * D_pred(q) + t
```

이 과정은 interpolation보다 geometry를 잘 보존하고, global alignment보다 local metric detail을 더 잘 반영한다.

---

#### 3) Fine Structure Refinement로 noise와 misalignment 보정

Coarse alignment 결과 `D_hat_prior`는 dense metric prior이지만, noisy measurement나 blurred edge에 민감할 수 있다.

이를 보정하기 위해 conditioned MDE model을 사용한다.

입력 condition은 다음과 같다.

```text
RGB image I
metric condition: D_hat_prior
geometry condition: D_pred
```

모델 구조상 RGB input layer와 병렬로 condition convolution layer를 추가한다. Condition layer는 zero initialization으로 시작하므로, 초기에는 pretrained MDE model의 능력을 해치지 않고 점진적으로 condition을 활용한다.

---

#### 4) Scale Normalization

`D_hat_prior`와 `D_pred`를 `[0, 1]` 범위로 normalize한다.

목적은 두 가지이다.

- Indoor/outdoor처럼 scene scale이 크게 다른 경우에도 일반화하기 위함
- Frozen MDE model을 test-time에 교체할 수 있도록 prediction scale 차이를 완화하기 위함

모델 출력은 다시 ground-truth scale로 de-normalization하여 loss를 계산한다.

---

#### 5) Synthetic Training Data 구성

실제 depth dataset은 blurred edge, missing value, sensor noise를 포함하므로, 논문은 정확한 GT를 가진 synthetic dataset을 활용한다.

사용 데이터셋:

- Hypersim
- vKITTI

정확한 GT depth에서 다음 synthetic prior를 생성한다.

- Sparse point sampling
- Square missing area
- Downsampling
- Outlier 추가
- Boundary noise 추가

이를 통해 depth completion, super-resolution, inpainting, mixed prior 상황을 학습 중에 모사한다.

---

#### 6) Learning Objective

Pixel-level supervision에는 ZoeDepth를 따라 **scale-invariant log loss**를 사용한다.

핵심 목적은 예측 depth와 GT depth의 log-space 차이를 줄이면서, depth scale 변화에 보다 안정적으로 학습하는 것이다.

---

### 출력

- 최종 출력은 dense하고 세밀한 **metric depth map**이다.

```text
D_output ∈ R^{H×W}
```

출력 특성:

- sparse / low-res / hole이 있는 prior를 dense depth로 복원
- MDE prediction의 fine geometry와 object boundary를 반영
- metric prior의 absolute scale을 유지
- noisy measurement와 blurred boundary를 일정 수준 보정
- depth completion, super-resolution, inpainting을 단일 모델로 처리

---

## 4. 메모

- 이 논문의 핵심은 **metric prior를 단순히 MDE에 prompt로 넣는 것**이 아니라, 먼저 MDE prediction을 이용해 prior를 dense한 intermediate domain으로 변환한 뒤, conditioned MDE가 이를 다시 refinement한다는 점이다.
- Coarse Metric Alignment는 parameter-free 단계이며, kNN + weighted least squares로 구현된다.
- Inference latency의 상당 부분은 kNN과 least squares 기반의 coarse alignment에서 발생한다.
- 실험에서는 7개 real-world dataset과 9개 prior pattern, 그리고 mixed prior를 사용해 zero-shot generalization을 평가한다.
- Mixed prior에서 PriorDA의 장점이 특히 두드러진다. 실제 센서 데이터는 sparse, low-resolution, missing area가 동시에 존재할 수 있기 때문이다.
- VGGT의 confidence 상위 30% depth를 prior로 사용해 depth를 refine하는 application도 제시한다.
- 로봇/실내 reconstruction 관점에서는 Kinect depth, LiDAR sparse depth, SLAM/TSDF 기반 pseudo depth, VGGT/MASt3R 계열 depth를 dense metric pseudo label로 보정하는 데 응용 가능하다.
- 다만 잘못된 metric prior가 강하게 들어가면 coarse alignment가 주변 영역을 오염시킬 수 있으므로, 실제 적용 시 reflective region 제거, edge outlier filtering, valid mask 정제가 중요하다.
- 사용자가 진행 중인 3D depth completion GT 생성 관점에서는, 완전한 GT라기보다 **metric prior + MDE geometry를 결합한 dense pseudo-GT 생성기**로 보는 것이 적절하다.
- 특히 Kinect RGB-aligned depth가 중앙 영역에만 존재하거나, LiDAR/SLAM depth가 sparse하게만 존재하는 경우에 실용적이다.

---

## 5. 적용 포인트
 - coarse dense metric depth 생성방법
