# Large Depth Completion Model from Sparse Observations

- **학회:** ICLR 2026 Poster
- **링크:** https://openreview.net/forum?id=I9o2OkPwCX
- **코드:** https://pkqbajng.github.io/ldcm/  
  - OpenReview 기준 코드와 모델 공개 페이지로 명시되어 있음. 다만 별도 접속 가능 여부는 확인 필요.
- **분야:** Depth Completion / Monocular Metric Depth Estimation / Point Map Estimation / 3D Geometry Estimation

---

## 1. 요약

- **LDCM**은 RGB 이미지와 sparse depth observation으로부터 dense metric depth map을 추정하는 depth completion framework이다.
- 기존 depth completion을 단순한 depth restoration, interpolation, denoising 문제로 보는 대신, **3D scene structure를 직접 예측하는 문제**로 재정의한다.
- 첫 번째 핵심은 monocular depth foundation model의 relative depth cue와 sparse metric depth를 결합해 **Poisson 기반 coarse dense depth map**을 생성하는 것이다.
- 두 번째 핵심은 기존 depth regression head 대신 **point map head**를 사용하여 각 픽셀의 camera-space 3D coordinate를 직접 회귀하는 것이다.
- 최종 depth map은 예측된 point map의 **z-channel**을 추출하여 얻는다.
- Poisson initialization은 sparse point가 매우 적거나 불규칙한 경우에도 relative depth의 구조적 gradient를 이용해 더 기하적으로 일관된 coarse depth를 만든다.
- Point map representation은 depth map보다 3D 구조를 명시적으로 표현하므로, metric consistency와 global geometry를 학습하는 데 유리하다.
- KITTI, ETH3D, iBims-1, DIODE, NYUv2, VOID 등 다양한 benchmark에서 zero-shot depth completion 성능을 평가한다.
- 실험 결과 LDCM은 depth completion, metric point map estimation, affine-invariant point map estimation 모두에서 기존 최신 방법보다 우수한 성능을 보인다.
- 특히 irregular sparse observation, extreme sparsity, unseen domain 조건에서 강한 generalization을 보이는 것이 핵심 장점이다.

---

## 2. 핵심 기여

- **Poisson 기반 coarse depth alignment 제안**
  - DepthAnythingV2-S와 같은 monocular depth foundation model의 relative depth cue를 sparse metric depth와 결합한다.
  - 단순 global affine alignment나 LWLR보다 sparse하고 불규칙한 observation에서 더 안정적인 coarse dense depth map을 생성한다.
  - Gradient-domain reconstruction을 통해 sparse point의 metric anchor와 relative depth의 fine geometric structure를 함께 활용한다.

- **Depth map regression을 point map regression으로 재정의**
  - 기존 방식처럼 per-pixel depth value를 직접 복원하지 않고, 각 픽셀의 camera-space 3D coordinate를 예측한다.
  - 모델이 2.5D depth restoration보다 더 명시적인 3D scene structure를 학습하도록 유도한다.
  - 최종 depth는 point map의 z-component로 얻는다.

- **Camera intrinsic 없이 metric-scale 3D geometry 출력**
  - Point map head가 metric 3D coordinate를 직접 예측하므로, 별도 camera intrinsic parameter 없이도 metric point map을 생성할 수 있다.
  - Uncalibrated environment에서의 deployment 가능성을 높인다.

- **강한 zero-shot 일반화 성능**
  - 다양한 sparse pattern, 데이터셋, indoor/outdoor 환경에서 기존 SOTA를 능가한다.
  - Depth completion뿐 아니라 point map estimation에서도 우수한 결과를 보인다.

---

## 3. 방법

### 입력

- **RGB image**
  - `I ∈ R^{H×W×3}`

- **Sparse depth map**
  - `S ∈ R^{H×W}`
  - 입력 sparse pattern은 random sampling, SfM keypoint, LiDAR-like structured sparsity 등 다양할 수 있다.

- **Monocular depth foundation model output**
  - Relative depth cue `D_r`
  - 논문 구현에서는 coarse alignment를 위해 **DepthAnythingV2-S**를 사용한다.

- **학습 시 ground-truth**
  - Dense metric depth map
  - Ground-truth metric point map `P̂`
  - Valid ground-truth mask `M`

---

### 핵심 아이디어

#### 1) Poisson 기반 coarse depth initialization

- LDCM은 먼저 sparse depth `S`와 foundation model의 relative depth `D_r`를 이용해 coarse dense depth map `C`를 만든다.
- 단순 sparse depth interpolation은 geometric prior가 약해 artifact가 생기기 쉽다.
- Global affine alignment는 전체 이미지에 하나의 scale/shift만 적용하므로 per-pixel metric structure를 복원하기 어렵다.
- LWLR은 local fitting을 통해 spatial adaptivity를 높이지만, sparse point가 매우 적거나 불규칙하면 성능이 불안정하다.
- LDCM은 이를 해결하기 위해 **Poisson reconstruction**을 사용한다.

핵심 optimization은 다음 형태이다.

```math
C = \arg\min_D
\left(
\sum_i \|\nabla \log D_i - \nabla \log(D_r + \gamma)_i\|^2
+
\lambda \sum_{i \in \Omega} (D_i - S_i)^2
\right)
```

- 첫 번째 항은 coarse depth의 log-gradient가 foundation model depth의 구조적 gradient와 일치하도록 만든다.
- 두 번째 항은 valid sparse point 위치에서 실제 sparse metric depth 값을 보존하도록 만든다.
- `Ω`는 valid sparse depth point 집합이다.
- `γ = β / α`이며, `α, β`는 relative depth `D_r`를 sparse depth `S`에 global affine alignment하기 위해 least squares로 구한 값이다.

이 방식의 의미는 다음과 같다.

- Sparse metric depth는 절대 scale을 anchor한다.
- Foundation model depth는 dense한 구조적 gradient를 제공한다.
- Poisson reconstruction은 이 둘을 gradient-domain에서 결합하여 metric-consistent coarse dense depth를 만든다.
- 결과적으로 sparse point가 적어도 geometry structure가 전체 이미지로 전파된다.

#### 2) ViT 기반 depth completion network

- 두 번째 단계에서는 RGB image `I`와 coarse depth `C`를 입력으로 사용한다.
- 네트워크는 dual encoder 구조를 사용한다.
  - RGB image encoder
  - Coarse depth encoder
- Feature fusion에는 PromptDA 계열의 **prompt fusion block**을 사용한다.
- Image encoder는 DINOv2로 pretraining된 **ViT-B**를 사용한다.

#### 3) Point map head

- 기존 depth completion network는 보통 dense depth map을 직접 회귀한다.
- LDCM은 depth head를 제거하고 **point map head**를 사용한다.
- Point map head는 각 픽셀에 대해 camera-space 3D coordinate를 예측한다.

```math
P \in R^{H \times W \times 3}
```

- 최종 dense depth map은 다음처럼 point map의 z-channel에서 얻는다.

```math
D = P_z
```

이 설계의 장점은 다음과 같다.

- Depth value만 맞추는 것이 아니라 3D geometry 자체를 학습한다.
- Local surface shape, global metric consistency, scene structure를 더 잘 반영한다.
- Camera intrinsic 없이 metric-scale point map을 직접 출력할 수 있다.

#### 4) Training losses

LDCM은 predicted point map `P`와 ground-truth point map `P̂` 사이에 세 가지 loss를 사용한다.

```math
L = L_{global} + \lambda_{local} L_{local} + \lambda_{normal} L_{normal}
```

- **Global point map loss**
  - 전체 point map 구조의 metric consistency를 맞춘다.
  - depth가 큰 영역의 scale imbalance를 줄이기 위해 `1 / D̂_i` 가중치를 사용한다.

- **Local point map loss**
  - 3D 공간에서 anchor point를 sampling하고 spherical neighborhood를 구성한다.
  - 이미지 plane 기준이 아니라 3D neighborhood 기준으로 local geometry coherence를 학습한다.

- **Normal loss**
  - 예측 point map과 GT point map에서 surface normal을 추정한다.
  - normal direction을 정렬하여 surface smoothness와 geometry alignment를 강화한다.

#### 5) 학습 설정

- 학습 데이터:
  - 11개 public RGB-D dataset
  - 약 270만 sample
  - indoor/outdoor scene 포함

- Backbone:
  - DINOv2 pretrained ViT-B image encoder

- Coarse depth foundation model:
  - DepthAnythingV2-S

- Optimizer:
  - AdamW

- Training schedule:
  - 200K iterations
  - cosine learning rate schedule
  - first 5% linear warmup

- Learning rate:
  - encoder: `1e-5`
  - other layers: `1e-4`

- Batch size:
  - global batch size 128

- Data augmentation:
  - random cropping
  - color jittering
  - Gaussian blur
  - JPEG compression-decompression
  - perspective-aware cropping

- Sparse depth generation:
  - dense GT depth를 다양한 pattern으로 subsampling
  - OMNI-DC protocol을 따름

- 학습 리소스:
  - 16 × H20 GPU
  - 약 6일

---

### 출력

- **Metric point map**
  - `P ∈ R^{H×W×3}`
  - 각 픽셀의 camera-space 3D coordinate

- **Dense metric depth map**
  - point map의 z-channel `P_z`를 추출하여 생성

- **Camera-intrinsic-free metric geometry**
  - 별도 camera intrinsic이 없어도 metric-scale 3D point map을 예측할 수 있음

---

## 4. 메모

- 이 논문의 핵심 관점은 depth completion을 **depth restoration 문제가 아니라 metric 3D geometry prediction 문제**로 보는 것이다.
- Poisson-based coarse alignment는 sparse metric depth와 foundation model relative depth를 결합하는 전처리 단계로 볼 수 있다.
- `∇ log(D_r + γ)`를 target gradient로 사용하는 이유는 relative depth의 unknown scale/shift 문제를 완화하면서 metric depth space에 더 잘 정렬된 gradient field를 만들기 위함이다.
- Sparse observation은 metric scale anchor 역할을 하고, monocular foundation model은 dense geometric structure prior 역할을 한다.
- Ablation 결과에서 Poisson alignment가 global affine alignment, LWLR, 단순 interpolation보다 좋은 coarse depth를 생성한다.
- 특히 LWLR은 sparse point가 충분하지 않거나 irregular하게 분포하면 local fitting이 불안정해질 수 있다.
- Output representation ablation에서는 depth map이나 depth + ray map보다 point map output이 더 좋은 성능을 보인다.
- 이는 point map이 camera ray + depth 조합보다 더 직접적으로 3D structure를 학습하도록 만들기 때문으로 해석할 수 있다.
- 평가 benchmark는 KITTI, ETH3D, iBims-1, DIODE, NYUv2, VOID 등을 포함한다.
- Sparse input pattern은 noisy random point, SIFT/ORB keypoint, LiDAR-simulated scan, VIO sparse map, COLMAP SfM point projection 등 다양하게 구성된다.
- LDCM은 depth completion뿐 아니라 point map estimation과 affine-invariant point map estimation에서도 좋은 성능을 보인다.
- Camera intrinsic이 없는 환경에서도 metric-scale 3D point map을 출력할 수 있다는 점은 robot, AR, uncalibrated camera deployment 측면에서 장점이다.
- 다만 실제 적용에서는 sparse depth noise, moving object, reflective surface, sensor calibration error, rolling shutter 등 real-world artifact에 대한 추가 검증이 필요하다.
- 로봇/실내 환경 관점에서는 sparse depth가 매우 적거나 SfM/VIO 기반으로 불규칙하게 들어오는 경우에 특히 유용할 가능성이 크다.
- Depth completion GT 생성 또는 dense metric depth pseudo-label 생성 파이프라인에서도 LDCM의 Poisson initialization + point map prediction 구조를 참고할 만하다.

---

## 5. 적용 포인트
 - coarse depth를 위한, possion initialization 방법과 point map loss 방법
