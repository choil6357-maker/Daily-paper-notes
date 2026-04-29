# Depth Completion as Parameter-Efficient Test-Time Adaptation (CAPA)

- **학회:** arXiv 2026 / TMLR under review
- **링크:** https://arxiv.org/abs/2602.14751
- **프로젝트:** https://research.nvidia.com/labs/dvl/projects/capa/
- **코드:** https://github.com/nv-dvl/capa
- **분야:** Depth Completion, Monocular Depth Estimation, 3D Foundation Model, Test-Time Adaptation, Parameter-Efficient Fine-Tuning, Video Depth Estimation

---

## 1. 요약

- CAPA는 sparse depth cue를 이용해 사전학습된 3D foundation model을 테스트 시점에서 장면별로 적응시키는 depth completion 방법이다.
- 기존 depth completion은 sparse depth를 추가 입력으로 받는 task-specific encoder를 학습하는 방식이 많았지만, 이러한 방식은 sensor pattern이나 sparsity pattern이 바뀌면 일반화가 약해질 수 있다.
- CAPA는 foundation model backbone을 고정하고, LoRA 또는 VPT와 같은 PEFT parameter만 업데이트한다.
- Sparse depth는 network input feature로 직접 넣는 것이 아니라, inference-time optimization을 위한 supervision 또는 geometric constraint로 사용된다.
- Base model의 raw depth prediction은 sparse depth와 scale/shift가 맞지 않을 수 있으므로, valid sparse point 위치에서 affine alignment를 수행한 뒤 loss를 계산한다.
- Video 또는 multi-view setting에서는 frame별로 따로 adaptation하지 않고, sequence 전체가 하나의 LoRA/VPT parameter set을 공유한다.
- 이 sequence-level parameter sharing은 여러 frame의 sparse cue를 통합하여 noisy condition point에 대한 robustness와 temporal consistency를 높인다.
- 실험에서는 VGGT를 주 base model로 사용하며, MoGe-2와 UniDepthV2에도 적용해 model-agnostic 특성을 보인다.
- ScanNet, 7-Scenes, iBims, Mapillary Metropolis 등 indoor/outdoor dataset에서 기존 depth completion 방법보다 낮은 error를 달성한다.
- CAPA는 특히 high-fidelity offline application, 예를 들어 3D mapping이나 pseudo-ground-truth generation에 적합한 접근으로 볼 수 있다.

---

## 2. 핵심 기여

- **Depth completion의 재정의**
  - Sparse depth를 직접 입력으로 넣는 task-specific depth completion network가 아니라, 사전학습된 3D foundation model을 테스트 시점에서 scene-specific하게 보정하는 문제로 재정의한다.
  - 즉, depth completion을 `dense prediction task`라기보다 `model adaptation task`로 바라본다.

- **PEFT 기반 test-time adaptation**
  - Foundation model backbone은 freeze하고, LoRA 또는 VPT parameter만 업데이트한다.
  - Sparse depth map에서 얻은 gradient를 이용해 매우 적은 수의 parameter만 최적화하므로, full fine-tuning보다 효율적이다.
  - VGGT 기준 LoRA rank `r = 4`, VPT prompt length `t = 16`을 사용하며, trainable parameter는 약 `0.39M`이다.

- **Video / multi-view를 위한 sequence-level parameter sharing**
  - 각 frame마다 별도 parameter를 최적화하지 않고, 하나의 sequence 전체가 동일한 PEFT parameter를 공유한다.
  - 여러 frame의 sparse geometric cue를 함께 사용하므로 noisy point에 덜 민감하고, frame 간 temporal consistency가 향상된다.
  - 긴 sequence에서는 매 optimization step마다 일부 frame만 sampling하는 mini-batch tuning을 사용한다.

---

## 3. 방법

### 입력

- **Single image setting**
  - RGB image: `I ∈ R^{H×W×3}`
  - Sparse depth map: `C ∈ R^{H×W}`
  - Valid sparse depth mask: `M ∈ R^{H×W}`

- **Video / multi-view setting**
  - RGB frame sequence: `(I_1, ..., I_N)`
  - Sparse depth sequence: `(C_1, ..., C_N)`
  - Valid mask sequence: `(M_1, ..., M_N)`

- **Base model**
  - 주로 VGGT를 사용한다.
  - CAPA에서는 VGGT의 ViT encoder, multi-view aggregator, depth head를 사용한다.
  - 다른 ViT 기반 3D foundation model에도 적용 가능하며, 논문에서는 UniDepthV2와 MoGe-2에도 적용한다.

---

### 핵심 아이디어

#### 1) Sparse depth를 입력이 아니라 optimization constraint로 사용

기존 depth completion 방식은 보통 다음과 같다.

```text
RGB image + sparse depth
    ↓
task-specific encoder
    ↓
dense depth
```

CAPA는 다음과 같이 접근한다.

```text
RGB image
    ↓
frozen 3D foundation model
    ↓
raw dense depth

sparse depth
    ↓
test-time loss / gradient
    ↓
LoRA or VPT parameter update
```

즉, sparse depth를 feature로 encoding하는 것이 아니라, foundation model의 prediction을 장면별 metric cue에 맞추기 위한 supervision으로 사용한다.

---

#### 2) LoRA 기반 adaptation

Transformer attention의 query, key, value projection matrix를 다음과 같이 low-rank update로 보정한다.

```text
W'_m = W_m + ΔW_m
ΔW_m = B_m A_m
m ∈ {q, k, v}
```

- 기존 projection weight `W_m`은 freeze한다.
- `A_m`, `B_m`만 trainable parameter로 둔다.
- Attention projection 자체를 직접 조정하므로, model feature response를 scene-specific하게 바꿀 수 있다.

---

#### 3) VPT 기반 adaptation

각 transformer layer의 token sequence 앞에 learnable visual prompt token을 붙인다.

```text
X_new = [P; X]
```

- `P`는 학습 가능한 prompt token이다.
- Prompt token이 attention distribution을 바꾸면서 image token representation에 영향을 준다.
- Attention 이후에는 image token만 유지하여 원래 token 길이를 보존한다.

---

#### 4) Sparse depth 기반 affine alignment

Foundation model의 raw depth prediction은 metric scale이 정확하지 않을 수 있다.  
따라서 raw prediction `d_hat`을 sparse depth `C`에 맞추기 위해 scale `s`와 shift `t`를 구한다.

```text
D_hat = s * d_hat + t
```

`s`, `t`는 valid sparse depth 위치에서 robust L1 minimization으로 계산한다.

```text
min_{s,t} || M ⊙ (s * d_hat + t - C) ||_1
```

이후 aligned depth `D_hat`과 sparse depth `C` 사이의 L1 loss를 valid pixel에서 계산하고, 이 loss를 backpropagation하여 LoRA/VPT parameter만 업데이트한다.

---

#### 5) Sequence-level adaptation

Video 또는 multi-view setting에서는 모든 frame이 하나의 shared PEFT parameter `θ`를 사용한다.

```text
θ_1 = θ_2 = ... = θ_N
```

이 방식의 장점은 다음과 같다.

- 여러 frame의 sparse depth cue를 통합할 수 있다.
- noisy condition point의 영향을 평균화할 수 있다.
- frame 간 scale이나 geometry가 흔들리는 문제를 줄일 수 있다.
- temporal consistency가 좋아진다.
- sequence마다 한 번만 adaptation하면 되므로 per-frame tuning보다 효율적이다.

긴 sequence에서는 매 step마다 일부 frame만 random sampling하여 mini-batch optimization을 수행한다.

---

#### 6) 전체 알고리즘 요약

```python
# Inputs:
# I: RGB image or video frames
# C: sparse depth map
# M: valid sparse depth mask
# F: pre-trained 3D foundation model
# theta: PEFT parameters, e.g. LoRA or VPT

freeze(F.backbone)
initialize(theta)

for step in range(num_steps):

    # video라면 일부 frame만 mini-batch로 sampling
    I_batch, C_batch, M_batch = sample_frames(I, C, M)

    # frozen FM + PEFT parameter로 raw dense depth 예측
    d_pred = F(I_batch, theta)

    # sparse depth에 맞게 scale/shift alignment
    s, t = solve_affine_alignment(d_pred, C_batch, M_batch)
    D_pred = s * d_pred + t

    # valid sparse point 위치에서 loss 계산
    loss = L1(M_batch * (D_pred - C_batch))

    # PEFT parameter만 업데이트
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 최종 dense depth 생성
D_final = F(I, theta)
D_final = affine_align(D_final, C, M)
```

---

### 출력

- **Single image**
  - Sparse depth cue와 일관되는 dense metric depth map `D_hat`

- **Video / multi-view**
  - Frame별 dense depth sequence `(D_hat_1, ..., D_hat_N)`
  - Sequence-level shared parameter를 사용하므로 temporal consistency가 향상된 depth prediction

- **Optional output**
  - Scene-specific LoRA/VPT parameter
  - Aligned dense depth
  - Sparse depth residual map
  - Pseudo-GT 생성용 confidence map과 함께 사용할 수 있는 dense depth

---

## 4. 메모

- CAPA의 가장 중요한 관점 변화는 sparse depth를 `입력 feature`가 아니라 `test-time optimization constraint`로 사용한다는 점이다.
- 기존 depth completion 모델은 학습 때 보지 못한 sparsity pattern, sensor pattern, range limitation에 취약할 수 있다.
- CAPA는 test sample 또는 sequence마다 직접 adaptation하므로, SIFT/SfM point, random sparse point, limited-range depth sensor, LiDAR scan-line pattern 등 다양한 condition pattern에 대응할 수 있다.
- 실험에서는 ScanNet, 7-Scenes, iBims, Mapillary Metropolis를 사용했고, indoor/outdoor 모두에서 성능을 검증했다.
- Depth metric은 주로 AbsRel을 사용하고, temporal consistency는 optical-flow-based warping loss인 OPW로 평가한다.
- CAPA_LoRA와 CAPA_VPT는 대부분의 setting에서 기존 baseline보다 우수하며, VGGT base model의 error를 약 2~3배 줄인다.
- Ablation 결과, video에서는 per-frame tuning보다 sequence-level adaptation이 depth accuracy와 temporal consistency 모두에서 더 좋다.
- Mini-batch tuning에서는 frame 비율을 늘리면 성능이 좋아지지만 약 10% 이후부터는 성능 향상이 포화된다.
- Optimization step은 약 100 step 이후 안정화되는 경향을 보인다.
- Full model fine-tuning이 가장 높은 성능을 보이긴 하지만, CAPA보다 훨씬 많은 parameter를 업데이트한다. CAPA는 훨씬 적은 parameter로 거의 비슷한 성능을 달성한다.


---

## 5. 적용 포인트
 - Noisy depth GT signal(ex, 라이다 센서로 부터 얻은 누적 PCD)에서 안정적인 full dense GT를 생성하는 용도로 활용할 수 있을것으로 예상
