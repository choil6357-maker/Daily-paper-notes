# Zero-shot Depth Completion via Test-time Alignment with Affine-invariant Depth Prior

- **학회:** AAAI 2025
- **링크:** https://arxiv.org/abs/2502.06338
- **코드:** https://github.com/kaist-ami/Zero-Shot-Depth-Completion
- **분야:** Depth Completion, Monocular Depth Estimation, Diffusion Model, Test-time Alignment, 3D Vision

---

## 1. 요약

- Sparse depth measurement로부터 dense metric depth map을 복원하는 **zero-shot depth completion** 방법을 제안한다.
- 기존 depth completion 방법은 주로 in-domain dense depth GT로 학습되어, out-of-domain 환경에서 depth affinity가 잘 일반화되지 않는 문제가 있다.
- 본 논문은 pre-trained **monocular depth diffusion model**을 depth prior로 사용한다.
- 이 depth prior는 **affine-invariant depth space**에서 동작하므로 metric scale은 없지만, 장면의 구조와 depth affinity를 잘 표현한다.
- Sparse metric depth는 실제 metric scale을 제공하지만, sparse하고 noisy하며 dense structure 정보가 부족하다.
- 제안 방법은 이 두 정보를 결합하기 위해 **test-time alignment**를 수행한다.
- Diffusion reverse sampling 과정 중 sparse depth를 **hard constraint**로 강제하여 affine-invariant prior를 metric depth space에 정렬한다.
- 또한 sparse depth outlier를 제거하기 위해 **prior-based outlier filtering**을 제안한다.
- 구조적 detail을 보존하기 위해 **R-SSIM loss**를 추가하여 overly smooth한 결과를 방지한다.
- 다양한 indoor/outdoor dataset에서 기존 방법 대비 우수한 domain generalization 성능을 보인다.

---

## 2. 핵심 기여

### 2.1 Zero-shot depth completion framework 제안

Target domain의 dense depth GT나 추가 training 없이, RGB image와 sparse metric depth만으로 dense metric depth를 복원한다.

### 2.2 Affine-invariant depth diffusion prior와 sparse metric depth의 test-time alignment

Monocular depth diffusion model의 구조적 prior를 사용하면서, sparse metric measurement를 hard constraint로 걸어 metric scale을 맞춘다.

### 2.3 Robust alignment를 위한 outlier filtering 및 structure-preserving loss 제안

Prior-based outlier filtering으로 noisy sparse depth를 제거하고, R-SSIM loss로 depth prior의 sharp structure를 유지한다.

---

## 3. 방법

### 입력

- RGB image $I$
- Sparse metric depth measurement $y$
  - 예: LiDAR, SLAM/VIO, RGB-D sensor 등에서 얻은 sparse depth
- Pre-trained monocular depth diffusion model
  - 예: Marigold, DepthFM

---

### 핵심 아이디어

Depth completion을 inverse problem으로 정의한다.

$$
A(x) = y
$$

여기서:

- $x$: unknown dense depth map
- $y$: sparse depth measurement
- $A$: dense depth에서 sparse measurement 위치만 선택하는 measurement operator

즉, dense depth $x$ 중에서 관측된 sparse point 위치만 골라낸 것이 $y$와 일치해야 한다.

MAP 관점에서는 다음과 같은 문제로 볼 수 있다.

$$
\arg\min_x \|y - A(x)\|_2^2 - \log p(x)
$$

여기서:

- $\|y - A(x)\|_2^2$: sparse depth와의 일치성
- $-\log p(x)$: depth prior를 따르도록 하는 regularization term

이때 $p(x)$에 해당하는 prior를 pre-trained depth diffusion model에서 가져온다.

---

### Diffusion prior를 이용한 guided sampling

일반적인 diffusion guided sampling은 다음과 같은 형태이다.

$$
\hat{s}_\theta(x_t, t, y)
=
s_\theta(x_t, t)
+
w \nabla_{x_t} L(f(x_0(x_t)), y)
$$

즉, diffusion model의 원래 score에 guidance loss의 gradient를 추가해서 sampling 방향을 조정한다.

이 논문에서는 sparse depth consistency를 guidance로 사용한다.

Latent diffusion model 기반 depth diffusion에서는 depth map을 직접 다루지 않고 latent $z$에서 처리한다.

$$
\hat{s}_\theta
=
s_\theta(z_t, t)
+
w \nabla_{z_t}
\left\|y - A(D(z_0(z_t)))\right\|_2^2
$$

여기서:

- $z_t$: timestep $t$의 noisy latent
- $z_0(z_t)$: $z_t$로부터 추정한 clean latent
- $D(\cdot)$: latent를 depth map으로 decoding하는 decoder
- $A(D(\cdot))$: 완성된 depth 중 sparse measurement 위치만 추출
- $y$: sparse metric depth

즉, diffusion sampling 중간마다 sparse depth와 맞도록 gradient를 넣는다.

---

### Test-time alignment with hard constraints

단순한 guidance 방식은 sparse depth와 가까워지도록 유도할 뿐, 반드시 sparse measurement와 일치한다는 보장은 없다.

그래서 논문은 **hard constraint correction step**을 추가한다.

기존 soft guidance 방식은 다음과 같이 볼 수 있다.

$$
z_t \leftarrow z_t + \text{gradient guidance}
$$

제안 방식은 다음과 같다.

1. 현재 latent $z_t$에서 clean latent $z_0(z_t)$를 추정한다.
2. $z_0(z_t)$를 sparse depth와 맞도록 직접 optimize한다.
3. Optimize된 $\hat{z}_0$에 다시 timestep $t$에 맞는 noise를 추가한다.
4. Noise level이 맞춰진 $\hat{z}_t$로 remapping한 뒤 diffusion reverse sampling을 계속 진행한다.

Optimization은 다음과 같다.

$$
\hat{z}_0(z_t)
=
\arg\min_{z_0(z_t)}
\left\|y - A(D(z_0(z_t)))\right\|_2^2
$$

이 과정은 현재 diffusion sample이 sparse metric depth와 확실히 일치하도록 correction하는 단계이다.

그 후, diffusion process의 noise level을 유지하기 위해 다시 noisy latent로 보낸다.

$$
p(\hat{z}_t \mid \hat{z}_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}\hat{z}_0,
(1-\bar{\alpha}_t)I
\right)
$$

여기서:

$$
\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i
$$

이고, $\alpha_t$는 timestep $t$에서의 variance schedule이다.

---

### 왜 $z_t$가 아니라 $z_0(z_t)$를 optimize하는가?

Pre-trained diffusion model은 timestep마다 정해진 noise distribution을 따르는 $z_t$를 입력으로 받는다.

그런데 $z_t$ 자체를 직접 optimize하면, 그 timestep의 noise 특성을 깨뜨릴 수 있다. 그러면 diffusion model이 기대하는 latent distribution에서 벗어나 suboptimal result가 나올 수 있다.

따라서 논문은 다음 방식을 사용한다.

- noisy latent $z_t$를 직접 최적화하지 않는다.
- $z_t$에서 추정한 clean latent $z_0(z_t)$를 optimize한다.
- 그 후 다시 timestep $t$의 noise level로 remapping한다.

이렇게 하면 sparse depth constraint를 만족하면서도 diffusion prior의 sampling process를 크게 깨지 않는다.

---

### Affine-invariant depth prior를 metric depth completion에 사용하는 이유

Pre-trained monocular depth diffusion model은 일반적으로 affine-invariant depth를 예측한다.

즉, depth 값은 다음과 같은 scale 및 shift ambiguity를 가진다.

$$
D_{\text{metric}} \approx aD_{\text{relative}} + b
$$

여기서 $a$와 $b$는 각각 scale과 shift를 의미한다.

따라서 다음과 같은 질문이 생긴다.

> Affine-invariant depth model을 metric depth completion에 써도 되는가?

논문은 empirical experiment를 통해 normalized metric depth도 depth diffusion model의 prior distribution 안에서 어느 정도 표현 가능하다고 주장한다.

따라서 affine-invariant prior를 sparse metric depth에 align하면 metric depth completion에 사용할 수 있다고 본다.

---

### Prior-based outlier filtering

실제 sparse depth는 완전히 신뢰할 수 없다.

예를 들면 다음과 같은 문제가 있을 수 있다.

- RGB와 depth가 synchronization되지 않은 경우
- 투명체나 반사체로 인한 see-through point
- LiDAR, SLAM, VIO의 noisy point
- 잘못 projection된 sparse point

이런 outlier가 있으면 test-time alignment가 잘못된 sparse depth를 hard constraint로 따라가게 되어 divergence나 성능 저하가 생길 수 있다.

이를 해결하기 위해 논문은 **monocular depth prior 기반 outlier filtering**을 수행한다.

절차는 다음과 같다.

1. Pre-trained depth model로 affine-invariant depth map $D_r$를 생성한다.
2. $D_r$를 superpixel 기반 local segment $S_i$로 분할한다.
3. 각 segment 안에서 sparse metric depth point $y_i$를 모은다.
4. Affine-invariant depth와 metric sparse depth 사이의 local linear relation을 fitting한다.
5. RANSAC을 사용해 outlier에 robust하게 fitting한다.
6. Fitting 결과와 크게 벗어나는 point를 outlier로 제거한다.
7. 남은 sparse point $y^*$만 test-time alignment에 사용한다.

즉, 전체 image에서 한 번에 outlier를 찾는 것이 아니라, depth prior가 비슷하다고 보는 local region 단위로 나누어 outlier를 제거한다.

---

### Loss 구성

Optimization objective는 세 가지 loss로 구성된다.

$$
L
=
L_{\text{depth}}
+
\lambda_{\text{smooth}}L_{\text{smooth}}
+
\lambda_{\text{r-ssim}}L_{\text{r-ssim}}
$$

#### Sparse depth consistency loss

Sparse metric depth와 prediction이 일치하도록 하는 loss이다.

$$
L_{\text{depth}}
=
\frac{1}{|\Omega(y)|}
\sum_{\Omega(y)}
\left|y - A(\hat{D})\right|
$$

역할:

- sparse measurement 위치에서 predicted depth가 실제 metric depth와 맞도록 한다.
- metric scale을 부여하는 핵심 loss이다.
- L1 loss를 사용해 outlier나 불확실성에 더 robust하게 만든다.

#### Local smoothness loss

Depth map이 local하게 smooth하도록 하는 regularization이다.

$$
L_{\text{smooth}}
=
\frac{1}{|\Omega|}
\sum_{c \in \Omega}
\lambda_X(c)
\left|\partial_X \hat{D}(c)\right|
+
\lambda_Y(c)
\left|\partial_Y \hat{D}(c)\right|
$$

여기서 RGB image gradient가 큰 edge 근처에서는 weight를 낮춘다.

$$
\lambda_X(c) = e^{-|\partial_X I(c)|}
$$

$$
\lambda_Y(c) = e^{-|\partial_Y I(c)|}
$$

역할:

- depth가 불필요하게 noisy해지는 것을 막는다.
- RGB edge 근처에서는 smoothness를 약하게 걸어 boundary를 보존한다.
- 기존 depth completion의 edge-aware smoothness와 유사한 역할을 한다.

#### Relative Structure Similarity Loss, R-SSIM

단순 smoothness만 사용하면 depth가 지나치게 부드러워지고, diffusion prior가 가진 sharp한 structure가 약해질 수 있다.

이를 막기 위해 논문은 R-SSIM loss를 제안한다.

$$
L_{\text{r-ssim}}(d_1, d_2)
=
1
-
\frac{
2\sigma_{d_1d_2} + C
}{
\sigma_{d_1}^{2}
+
\sigma_{d_2}^{2}
+
C
}
$$

기존 SSIM에서 luminance term을 제거한 형태이다.

이유는 다음과 같다.

- Relative depth와 metric depth는 absolute value range가 다르다.
- 따라서 absolute intensity/value를 직접 비교하면 안 된다.
- 대신 local structure, contrast, correlation 중심으로 비교해야 한다.

역할:

- off-the-shelf relative depth model이 가진 구조 정보를 completed metric depth에 전달한다.
- edge, object boundary, thin structure, local shape를 보존한다.
- overly smooth depth completion을 방지한다.

---

### 출력

- Dense metric depth map
- Sparse depth 위치에서는 metric measurement와 일치하고, 미관측 영역에서는 diffusion depth prior를 따라 자연스럽고 구조적인 depth를 생성한 결과

---

## 4. 메모

- 이 방법의 핵심은 **metric scale은 sparse depth에서 가져오고, dense structure는 monocular depth diffusion prior에서 가져오는 것**이다.
- 논문에서 말하는 zero-shot은 “test-time optimization이 전혀 없다”는 뜻이 아니다.
- 더 정확히는 **target-domain training 없이 test sample 단위로 latent를 최적화하는 방식**에 가깝다.
- Model weight를 fine-tuning하지 않고 latent/sample을 optimize하기 때문에, 일반적인 one-shot/few-shot domain adaptation과는 다르다.
- 장점은 domain generalization과 구조적 detail 보존이다.
- 단점은 diffusion sampling과 test-time optimization 때문에 inference가 느릴 수 있다는 점이다.
- 실용적으로는 CAPA 같은 parameter-efficient TTA 방식보다 느릴 가능성이 크지만, sparse metric depth와 monocular depth prior를 정렬하는 아이디어는 depth completion GT 생성이나 SLAM/LiDAR 기반 dense depth refinement에 참고할 만하다.

---

## 전체 알고리즘 흐름

```text
Input:
  RGB image I
  sparse metric depth y

1. Pre-trained monocular depth diffusion model로 depth prior 준비
   - affine-invariant depth prior 사용

2. Prior-based outlier filtering
   - relative depth Dr 추정
   - Dr 기반 superpixel segment 생성
   - 각 segment에서 sparse metric depth와 relative depth를 RANSAC으로 fitting
   - deviation이 큰 sparse point 제거
   - filtered sparse depth y* 생성

3. Diffusion reverse sampling 시작
   - latent z_t에서 sampling 진행

4. 일정 interval마다 test-time alignment 수행
   a. z_t에서 clean latent z_0(z_t) 추정
   b. decoder D로 completed depth D_hat 생성
   c. sparse depth 위치에서 y*와 D_hat이 맞도록 z_0 optimize
   d. optimize된 z_0에 timestep t에 맞는 Gaussian noise 추가
   e. 다시 z_t로 remapping

5. Optimization loss 적용
   - sparse depth consistency
   - edge-aware local smoothness
   - R-SSIM structural regularization

6. Reverse sampling 종료 후 dense metric depth map 출력
```

---

## 5. 적용 포인트
 - 반사 물체가 depth align에 사용되지 않도록 local로 나눠서 RANSAC을 이용한 fitting을 수행하는 루틴
 - R-SSIM loss와 local smoothness loss
