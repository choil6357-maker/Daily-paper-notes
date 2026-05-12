# Uni-SLAM: Uncertainty-Aware Neural Implicit SLAM for Real-Time Dense Indoor Scene Reconstruction

- **학회:** WACV 2025
- **링크:** https://shaoxiang777.github.io/project/uni-slam/
- **논문:** https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Uni-SLAM_Uncertainty-Aware_Neural_Implicit_SLAM_for_Real-Time_Dense_Indoor_Scene_WACV_2025_paper.pdf
- **코드:** https://github.com/dfki-av/Uni-SLAM
- **분야:** Dense RGB-D SLAM, Neural Implicit Representation, Indoor 3D Reconstruction, Uncertainty-aware SLAM

---

## 1. 요약

- **Uni-SLAM**은 RGB-D 입력을 사용하는 real-time dense implicit SLAM 방법이다.
- 기존 NeRF/implicit SLAM 방법들은 입력 RGB-D 데이터의 품질 차이를 충분히 고려하지 않고, 보통 매 `n` frame마다 고정적으로 mapping을 수행한다.
- Uni-SLAM은 입력 데이터의 품질이 frame/pixel마다 다르다는 점에 주목하고, 이를 **predictive uncertainty**로 모델링한다.
- 핵심 구조는 **geometry hash grid**와 **appearance hash grid**를 분리한 decoupled scene representation이다.
- Geometry와 color를 하나의 representation에 섞지 않고 따로 학습함으로써 thin structure와 high-frequency detail을 더 안정적으로 복원한다.
- Volume rendering 과정에서 ray의 **termination probability**를 계산하고, 이를 기반으로 pixel-level uncertainty와 image-level uncertainty를 정의한다.
- Pixel-level uncertainty는 tracking/mapping loss reweighting에 사용되어 invalid depth, motion blur, pose error 등의 영향을 줄인다.
- Image-level uncertainty는 추가 mapping, local BA, local loop closure optimization을 활성화하는 기준으로 사용된다.
- 실험은 Replica, ScanNet, TUM RGB-D에서 수행되었으며, neural implicit SLAM 계열 대비 tracking accuracy와 reconstruction quality를 개선한다.
- 특히 Replica dataset에서 depth L1 error를 25% 줄이고, 1cm 기준 completion ratio 66.86%를 달성했다고 보고한다.

---

## 2. 핵심 기여

- **Predictive uncertainty 기반 loss reweighting**
  - Ray termination probability를 이용해 별도 uncertainty network 없이 pixel-level predictive uncertainty를 정의한다.
  - Uncertainty가 높은 pixel은 tracking loss에서 제외하거나 약화하여 pose optimization을 안정화한다.

- **Uncertainty-guided strategic BA**
  - 고정 주기 mapping만 수행하지 않고, image-level uncertainty가 높을 때 추가 local BA를 활성화한다.
  - Co-visibility 기반 keyframe selection과 local loop closure optimization(LLCO)을 통해 local-to-global 정보를 균형 있게 반영한다.

- **Decoupled hash-grid scene representation**
  - Geometry와 appearance를 각각 별도의 multiresolution hash grid로 표현한다.
  - 기존 coupled representation보다 학습 난이도를 낮추고, thin structure와 high-frequency detail 복원에 유리하다.

---

## 3. 방법

### 입력

- RGB-D image sequence
- Camera intrinsic parameters
- Frame-wise camera pose estimate
- Depth-guided ray sampling을 위한 depth image
- Keyframe database

### 핵심 아이디어

#### 1) Decoupled Neural Scene Representation

Uni-SLAM은 scene을 geometry와 appearance로 분리하여 표현한다.

- Geometry hash grid: `h_g(x_i)`
- Appearance hash grid: `h_a(x_i)`
- Geometry decoder: `f_g`
- Appearance decoder: `f_a`

수식은 다음과 같다.

```math
\Phi_g(x_i) = f_g(h_g(x_i))
```

```math
\Phi_a(x_i) = f_a(h_a(x_i))
```

여기서 `\Phi_g(x_i)`는 raw SDF, `\Phi_a(x_i)`는 raw color이다.

이 구조는 geometry와 color가 동일한 spatial frequency로 표현될 필요가 없다는 가정에 기반한다. 따라서 geometry reconstruction과 appearance learning을 분리하여 thin object, edge, chair leg, table leg 같은 구조를 더 잘 복원하도록 설계되어 있다.

#### 2) SDF 기반 Volume Rendering

SDF는 density로 변환되고, density는 ray 위 sample point의 rendering weight를 계산하는 데 사용된다.

```math
w_i =
\exp\left(
-\sum_{j=1}^{i-1}\sigma(x_j)
\right)
\left(
1-\exp(-\sigma(x_i))
\right)
```

Color와 depth는 다음과 같이 rendering된다.

```math
\hat{c} = \sum_{i=1}^{N} w_i \Phi_a(x_i)
```

```math
\hat{d} = \sum_{i=1}^{N} w_i d_i
```

즉, Uni-SLAM은 SDF를 직접 depth로 변환하지 않고, `SDF → density → volume rendering weight → rendered color/depth` 흐름을 사용한다.

#### 3) Predictive Uncertainty

Ray `r`의 accumulated termination probability는 다음과 같이 정의된다.

```math
p(r) = \sum_{i=1}^{N} w_i
```

또는 다음과 같이 쓸 수 있다.

```math
p(r) =
1 - \exp\left(
-\sum_{i=1}^{N}\sigma(x_i)
\right)
```

해석은 다음과 같다.

- `p(r) ≈ 1`: 이미 잘 관측된 영역이며 model confidence가 높음
- `p(r) ≈ 0`: unobserved region이거나 invalid depth/motion blur/pose error 가능성이 높음

Pixel-level uncertainty는 다음과 같다.

```math
\beta_m = (1 - p(r_m))^2
```

Image-level uncertainty는 image 내 sampled ray들의 평균으로 정의된다.

```math
\beta =
\frac{1}{M}
\sum_{m=1}^{M}
\beta_m
```

#### 4) Uncertainty-guided Loss Reweighting

Pixel-level binary confidence function은 다음과 같다.

```math
CF_m =
\begin{cases}
1 & \text{if } \beta_m \le \beta^{unc}_m \\
0 & \text{if } \beta_m > \beta^{unc}_m
\end{cases}
```

즉, uncertainty가 threshold보다 낮은 pixel만 신뢰도 높은 pixel로 사용한다.

Tracking loss는 다음과 같다.

```math
L_t =
\lambda_{rgb}L^{track}_{rgb}
+
\lambda_{dep}L_{dep}
+
L_{sdf}
```

Tracking에서는 scene representation을 고정하고 camera pose만 optimize한다. 따라서 uncertainty가 높은 pixel은 pose optimization에 악영향을 줄 수 있으므로 loss에서 약화하거나 제외한다.

Mapping loss는 다음과 같다.

```math
L_m =
\lambda_{rgb}L^{map}_{rgb}
+
\lambda_{dep}L_{dep}
+
L_{sdf}
```

Mapping에서는 scene representation을 업데이트한다. 이때 RGB 정보는 invalid depth를 보완할 수 있기 때문에, mapping의 RGB loss에는 confidence function을 적용하지 않는다.

#### 5) Strategic Bundle Adjustment

기본적으로 Uni-SLAM은 다음과 같이 동작한다.

- Tracking: every frame
- Global BA 포함 mapping: every `n` frames
- Additional local BA: image-level uncertainty가 높을 때 활성화
- LLCO: co-visibility 기반 loop closure가 감지될 때 활성화

Image-level uncertainty가 threshold보다 크면 추가 mapping/local BA를 수행한다.

```math
\beta > \beta_{unc}
```

Local BA에서는 current frame과 시각적으로 overlap되는 keyframe만 선택한다. Co-visibility overlap coefficient는 다음과 같다.

```math
OC_{cov}(i,c)
=
\frac{|I_i \cap I_c|}{|I_c|}
```

논문에서는 `OC_cov > 0.95`이면 loop closure로 판단하고, current frame부터 loop closure point까지의 keyframe만 최적화하는 LLCO를 수행한다.

### 출력

- Estimated camera trajectory
- Dense implicit scene representation
- Rendered RGB image
- Rendered depth map
- TSDF/SDF 기반 geometry
- Reconstructed mesh
- Pixel-level uncertainty map
- Image-level uncertainty score
- Keyframe database 및 BA 결과

---

## 4. 메모

- Uni-SLAM의 핵심은 “모든 pixel/frame을 동일하게 믿지 않는다”는 점이다.
- Tracking에서는 high-confidence pixel만 사용하여 pose estimation을 안정화한다.
- Mapping에서는 RGB 정보를 더 적극적으로 사용하여 invalid depth의 약점을 보완한다.
- Predictive uncertainty는 별도 학습 network 없이 volume rendering의 termination probability에서 직접 계산된다.
- Decoupled hash grid는 geometry와 appearance의 복잡도가 다르다는 점을 반영한 구조이다.
- Strategic BA는 고정 주기 mapping의 한계를 줄이고, uncertainty가 높은 frame에서 local reconstruction을 보강한다.
- Co-visibility 기반 LLCO는 global BA보다 계산 부담이 낮으면서 loop closure 상황에서 stability를 높이는 역할을 한다.
- 실내 환경에서 얇은 구조물, 예를 들어 의자 다리, 테이블 다리, 얇은 가구 edge를 복원하는 데 강점을 보인다.
- 코드 기준으로 Replica, ScanNet, TUM RGB-D 실행 및 ATE/reconstruction evaluation script가 제공된다.
- Custom RGB-D data에 적용하려면 RGB/depth frame, intrinsic, pose initialization 또는 tracking 설정, dataset config 구성이 필요하다.

---

## 5. 적용 포인트
 - uncertainty를 weight로 변환하는 방법
