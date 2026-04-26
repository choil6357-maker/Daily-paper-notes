# 제목: A Simple yet Universal Framework for Depth Completion

- **학회:** 미확인
- **링크:** 미확인
- **코드:** https://github.com/JinhwiPark/UniDC
- **분야:** Depth Completion, Monocular Depth Estimation, Foundation Model, Few-shot Learning, Sensor-/Domain-agnostic Depth Completion, Hyperbolic Geometry

---

## 1. 요약

- 이 논문은 다양한 장면과 다양한 depth sensor 환경에서도 일관되게 동작하는 **Universal Depth Completion(UniDC)** 문제를 정의한다.
- 기존 depth completion 방법들은 KITTI의 Velodyne LiDAR, NYUv2의 Kinect처럼 특정 센서와 특정 도메인에 최적화되는 경우가 많아, 새로운 센서나 새로운 환경으로 일반화하기 어렵다.
- 제안 방법은 sparse depth를 직접 CNN으로 completion하는 대신, **monocular depth foundation model**이 가진 상대적 3D 구조 이해 능력을 활용한다.
- Foundation model은 RGB 이미지에서 relative depth-aware feature와 relative depth map을 추출하며, 이 정보를 이용해 임의 센서에서 얻은 sparse metric depth를 dense depth로 전파한다.
- Baseline architecture는 크게 `foundation feature extraction → sparse-to-dense conversion → CSPN++ refinement`의 세 단계로 구성된다.
- Advanced architecture는 여기에 **hyperbolic geometry**를 도입하여 3D depth data의 계층적 구조를 더 잘 표현하도록 한다.
- Hyperbolic space는 sparse depth point를 중심으로 같은 surface/object에 속하는 픽셀은 가깝게, 다른 depth layer나 background에 속하는 픽셀은 멀게 표현하는 데 유리하다.
- 논문은 few-shot 및 zero-shot setting에서 제안 방법이 기존 depth completion 방법보다 뛰어난 sensor/domain adaptation 성능을 보인다고 주장한다.
- 실험은 NYUv2, KITTI DC, SUN-RGBD 등 다양한 dataset과 sensor 환경에서 수행되었다.
- 한계로는 현재 입력이 단일 RGB 이미지와 대응되는 sparse depth pair에 한정되며, multi-view 3D reconstruction이나 radar와 같은 noisy sparse modality로의 확장은 향후 연구로 남아 있다.

---

## 2. 핵심 기여

- **Universal Depth Completion 문제 정의**
  - 기존 depth completion이 특정 센서와 도메인에 강하게 의존한다는 한계를 지적하고, 임의의 센서와 임의의 환경에 적응 가능한 UniDC 문제를 정의하였다.
  - 특히 dense GT depth를 대규모로 확보하기 어려운 현실적인 상황을 고려하여, few-shot 또는 minimal labeled data setting에서의 일반화를 목표로 한다.

- **Depth foundation model 기반의 sensor-/domain-agnostic baseline 제안**
  - 별도의 depth encoder를 학습하지 않고, monocular depth foundation model에서 추출한 depth-aware feature를 사용한다.
  - 이를 통해 sparse depth의 sampling pattern, sensor scan range, indoor/outdoor domain 차이에 대한 overfitting을 줄인다.
  - Sparse depth를 직접 CNN에 넣어 completion하는 대신, foundation model feature를 이용해 sparse metric depth를 dense initial depth로 전파한다.

- **Hyperbolic geometry를 이용한 advanced architecture 제안**
  - 3D scene structure와 sparse depth propagation의 계층적 관계를 더 잘 표현하기 위해 hyperbolic embedding을 도입하였다.
  - Scene-dependent curvature generator와 multi-curvature affinity map을 통해 kernel size와 scene structure에 맞는 hyperbolic space를 동적으로 구성한다.
  - Hyperbolic feature 기반 propagation과 refinement를 통해 bleeding error를 줄이고, 적은 데이터에서도 일반화 성능을 높인다.

---

## 3. 방법

### 입력

- **RGB 이미지**
  - 입력 이미지 `I ∈ R^{3×H×W}`.
  - Monocular depth foundation model의 입력으로 사용된다.

- **Sparse depth map**
  - LiDAR, ToF, Kinect, RealSense 등 임의의 active depth sensor에서 얻은 sparse 또는 low-resolution depth.
  - Metric scale 정보를 포함한다.
  - 센서에 따라 density, scan range, noise pattern이 다르다.

- **Camera parameter**
  - BPNet/SPN 계열 방법과 유사하게 3D point 또는 spatial relation 계산에 사용될 수 있다.

- **선택적 dense GT depth**
  - Supervised few-shot setting에서는 dense GT depth를 사용한다.
  - Dense GT가 없는 더 현실적인 setting에서는 8-line/32-line LiDAR 입력을 사용하고 64-line LiDAR를 supervision으로 사용하는 self-supervised 형태도 실험한다.

---

### 핵심 아이디어

#### 1) Depth completion을 직접 회귀 문제가 아니라 propagation 문제로 재정의

기존 방법은 RGB 이미지와 sparse depth를 concatenate하여 CNN encoder-decoder로 dense depth를 직접 예측하는 경우가 많다. 하지만 이 방식은 특정 sensor pattern과 domain에 쉽게 overfit된다.

이 논문은 sparse depth를 직접 CNN에 넣기보다, **이미 3D 구조를 잘 알고 있는 monocular depth foundation model의 feature를 이용해 sparse metric depth를 dense하게 전파**한다.

전체 baseline 흐름은 다음과 같다.

```text
RGB image
  ↓
Monocular Depth Foundation Model
  ↓
Relative depth-aware feature + relative depth map
  ↓
Sparse-to-dense conversion
  ↓
Initial dense metric depth
  ↓
CSPN++ refinement
  ↓
Final dense metric depth
```

---

#### 2) Foundation model feature extraction

입력 이미지 `I`에 대해 pre-trained depth foundation model `f_F`를 사용한다.

```math
E, D_{relative} = f_F(I, Θ_{f_F})
```

- `E`: multi-scale intermediate feature
- `D_relative`: relative depth map
- `Θ_fF`: foundation model parameter

Foundation model은 metric depth를 직접 맞추기보다는 장면의 상대적 3D 구조, foreground/background 관계, object boundary, depth discontinuity를 잘 포착한다. 이 정보를 sparse depth propagation의 기준으로 사용한다.

---

#### 3) Foundation model tuning

Foundation model은 기본적으로 relative depth를 예측하므로 metric depth와 scale mismatch가 발생한다. 이를 줄이기 위해 scale-invariant loss를 사용한다.

```math
δ_v = log D_{relative}(v) - log D_{gt}(v)
```

```math
L_{scale-invariant}
=
\frac{1}{|V|}
\sum_{v∈V} (δ_v)^2
-
\frac{λ}{|V|^2}
\left(
\sum_{v∈V} δ_v
\right)^2
```

- 논문에서는 `λ = 0.85`를 사용한다.
- 전체 backbone을 fine-tuning하지 않고 **bias tuning**을 적용한다.
- Bias term만 업데이트하고 나머지 backbone parameter는 고정하여 foundation model의 high-resolution detail과 contextual information을 보존한다.

---

#### 4) Sparse-to-dense conversion

임의 센서에서 얻은 sparse depth `S`를 dense initial depth `D_init`로 변환한다.

기본 아이디어는 bilateral filtering과 유사하다.

```math
D_i^{init} = \sum_j w_{ij} S_j
```

여기서 `w_ij`는 pixel `i`에 대해 주변 sparse depth `S_j`를 얼마나 반영할지 결정하는 propagation weight이다.

기존 bilateral filter는 radiometric difference와 spatial distance를 함께 사용한다.

```math
w_{ij} = f_r(x_j, x_i) g_s(x_j - x_i)
```

이 논문에서는 RGB 차이만 사용하는 것이 아니라, foundation model에서 얻은 depth-aware feature를 사용해 sparse depth를 더 정확하게 전파한다.

---

#### 5) CSPN++ 기반 refinement

Sparse-to-dense conversion으로 얻은 initial dense depth를 CSPN++ 기반 spatial propagation으로 refinement한다.

```math
K = \{3, 5, 7\}
```

3×3, 5×5, 7×7 multi-kernel affinity map을 사용하며, propagation은 다음과 같은 형태로 수행된다.

```math
\hat{D}_{i}^{t+1}
=
\sum_{k∈K}
σ_{i,k}D_{i,k}^{t+1}
```

```math
D_{i,k}^{t+1}
=
A_{i,k} \odot D_i^0
+
\sum_{j∈N_k(i)}
A_{j,k} \odot D_{j,k}^{t}
```

- `D^0`: initial dense depth
- `A`: affinity map
- `N_k(i)`: kernel size `k` 내의 neighboring pixel set
- `σ`: kernel별 confidence map
- `D^t`: propagation step `t`에서의 depth map

이 단계는 local consistency를 강화하고 object boundary 주변의 depth를 정교하게 보정한다.

---

#### 6) Hyperbolic geometry 기반 advanced architecture

Advanced architecture는 baseline에 hyperbolic geometry를 추가한다.

목적은 다음과 같다.

- 3D scene의 계층 구조를 더 잘 표현
- Sparse depth propagation에서 bleeding error 감소
- Few-shot/zero-shot 상황에서 일반화 성능 향상
- Sensor/domain shift에 더 강한 pixel relation 학습

전체 흐름은 다음과 같다.

```text
RGB image
  ↓
Depth foundation model
  ↓
Multi-scale intermediate features
  ↓
Multi-scale feature fusion
  ↓
Curvature generator
  ↓
Euclidean feature → Hyperbolic feature
  ↓
Hyperbolic sparse-to-dense propagation
  ↓
Initial dense metric depth
  ↓
Multi-curvature hyperbolic affinity map
  ↓
CSPN++ refinement
  ↓
Final dense metric depth
```

---

#### 7) Multi-scale feature fusion

Foundation model에서 얻은 multi-scale intermediate feature를 결합한다.

```math
E_{l+1}^{M} = f_l^{fusion}(E_l^M, E_{l+1})
```

- Coarse feature를 upsample한 뒤 finer feature와 결합한다.
- `f_l^{fusion}`은 transposed convolution과 skip connection으로 구성된다.
- 목적은 low-level detail과 high-level context를 모두 포함한 depth-aware feature를 만드는 것이다.

---

#### 8) Hyperbolic embedding

Fused feature `E_L^M`을 hyperbolic space로 보낸다.

```math
H_i = exp_0^κ(E_{L,i}^{M})
```

- `H_i`: pixel `i`의 hyperbolic feature
- `κ`: hyperbolic curvature
- `E_{L,i}^{M}`: fused Euclidean feature

Hyperbolic space는 Euclidean space보다 tree-like 또는 hierarchical relationship을 표현하는 데 유리하다. Depth scene에서는 foreground/background, object/surface, sparse point 주변의 propagation hierarchy가 자연스럽게 존재하므로 hyperbolic embedding이 도움이 된다.

---

#### 9) Curvature generation

기존 hyperbolic method는 curvature `κ`를 고정 hyperparameter로 두는 경우가 많다. 하지만 UniDC에서는 scene, sensor, sparse depth pattern이 계속 달라지므로 고정 curvature가 항상 적합하지 않다.

따라서 논문은 scene-dependent curvature generator를 사용한다.

```math
κ = C(E_L^M)
```

Curvature generator는 convolution layer, MLP, global mean pooling으로 구성된다.

---

#### 10) Hyperbolic sparse-to-dense conversion

Hyperbolic feature를 이용해 sparse depth propagation weight를 계산한다.

```math
D_i^{init}
=
\sum_j w_{ij} S_j
```

```math
w_{ij}
=
P(
Dist_{hyp}(H_i, H_j),
Dist_{euc}(E_{L,i}^{M}, E_{L,j}^{M})
)
```

- `Dist_hyp`: hyperbolic feature distance
- `Dist_euc`: Euclidean distance
- `P`: learnable MLP
- `S_j`: sensor sparse depth
- `D_i^{init}`: pixel `i`의 initial dense depth

이 방식은 단순히 가까운 픽셀로 depth를 퍼뜨리는 것이 아니라, 같은 object/surface/depth layer에 속할 가능성이 높은 픽셀로 depth를 전파한다.

---

#### 11) Hyperbolic Convolution Layer, HCL

Refinement 단계에서 affinity map을 hyperbolic space에서 생성하기 위해 HCL을 사용한다.

```math
HCL(h, κ)
:=
W \otimes_κ T_β(h) \oplus_κ b
```

- `W`: convolution weight
- `b`: bias
- `⊗_κ`: hyperbolic multiplication
- `⊕_κ`: hyperbolic addition
- `T_β`: hyperbolic concatenation

일반 convolution을 hyperbolic geometry 연산으로 확장한 형태라고 볼 수 있다.

---

#### 12) Multi-curvature affinity generation

CSPN++의 multi-kernel refinement에 맞춰 kernel size별로 다른 curvature를 생성한다.

```math
κ_k = C_k(E_L^M)
```

```math
A_k^{hyp} = HCL(E_{L,i}^{M}, κ_k)
```

- 3×3 kernel: local edge/detail 중심
- 5×5 kernel: object part 또는 중간 범위 relation
- 7×7 kernel: 넓은 surface/context relation

Kernel size별로 다른 curvature를 사용함으로써, 서로 다른 receptive field에 적합한 pixel relation을 학습한다.

---

#### 13) Loss function

전체 학습 loss는 final dense depth를 위한 L1+L2 loss와 relative-to-metric scale gap을 줄이기 위한 scale-invariant loss로 구성된다.

```math
L =
L_{L1L2}(\hat{D}, D_{gt})
+
μ L_{scale-invariant}(D_{relative}, D_{gt})
```

```math
L_{L1L2}
=
\frac{1}{|V|}
\sum_{i∈V}
\left(
|\hat{D}_i - D_{gt,i}|
+
|\hat{D}_i - D_{gt,i}|^2
\right)
```

- 논문에서는 `μ = 0.1`을 사용한다.

---

### 출력

- **Final dense metric depth map**
  - 입력 RGB 이미지와 동일한 해상도의 dense depth prediction.
  - Sparse sensor depth의 metric scale을 유지하면서 이미지 전체로 확장된 depth map이다.

- **Intermediate output**
  - `D_relative`: foundation model의 relative depth map
  - `E`: multi-scale intermediate feature
  - `D_init`: sparse-to-dense conversion으로 얻은 initial dense depth
  - `A_k`: CSPN++ refinement에 사용되는 affinity maps
  - `A_k^{hyp}`: advanced architecture에서 생성되는 hyperbolic affinity maps
  - `κ_k`: kernel size별 scene-dependent curvature

---

## 4. 메모

- 이 논문의 핵심은 sparse depth를 CNN에 직접 넣어 completion하는 것이 아니라, **monocular depth foundation model이 가진 장면 구조 지식을 이용해 sparse metric depth를 전파**하는 것이다.
- Foundation model은 metric scale에는 약하지만, foreground/background, object boundary, relative depth ordering 같은 구조 정보를 잘 제공한다.
- UniDC의 baseline은 별도 depth encoder를 학습하지 않기 때문에 센서 scan pattern이나 domain-specific sparse depth distribution에 덜 overfit될 가능성이 있다.
- Hyperbolic geometry는 3D scene의 계층적 구조를 표현하기 위한 장치로 사용된다.
- Depth completion 관점에서 hyperbolic space의 역할은 “어디까지 depth를 퍼뜨리고 어디서 멈출지”를 더 잘 결정하는 것이다.
- Hyperbolic propagation은 RGB similarity만으로 구분하기 어려운 foreground/background 경계에서 bleeding error를 줄이는 데 도움이 될 수 있다.
- Ablation study에 따르면 hyperbolic embedding은 zero-shot/few-shot setting에서 Euclidean feature보다 더 나은 성능을 보였고, few-shot regime에서 평균 약 5% 성능 향상이 보고되었다.
- Feature fusion을 제거하면 특히 1-shot scenario에서 성능이 크게 감소하는 것으로 나타났다.
- Foundation model의 fine-tuning strategy, 특히 bias tuning은 relative depth와 metric depth 사이의 discrepancy를 줄이기 위해 중요하다.
- Foundation backbone으로 MiDaS, Depth Anything, UniDepth 등을 비교했으며, 논문에서는 MiDaS가 10-shot 및 100-shot scenario에서 더 적합하다고 주장한다.
- 이 결과는 convolution의 local inductive bias가 depth completion의 local propagation 구조와 잘 맞기 때문으로 해석된다.
- Dense GT depth 없이 sparse LiDAR setting에서 self-supervised 방식으로도 학습 가능성을 보였으며, 이는 실제 outdoor 환경에서 중요하다.
- 한계는 현재 single-view RGB + sparse depth pair만 입력으로 사용한다는 점이다.
- Multi-view reconstruction, novel view synthesis, radar와 같은 noisy/highly sparse modality로 확장하려면 uncertainty modeling이 추가로 필요하다.
- 코드 공개 repo에는 hyperbolic 연산 관련 모듈은 존재하지만, 확인 당시 핵심 model file 일부가 비어 있어 논문 전체 architecture가 완전하게 공개되어 있는지는 추가 확인이 필요하다.

---

## 5. 적용 포인트
 - 초기 inital depth를 생성하기 위해, Foundation model의 multi scale feature를 hyperbolic space에서 bilateral 방식으로 생성
 - Affinity map을 이용한 update에 hyperbolic space를 이용한 점.
