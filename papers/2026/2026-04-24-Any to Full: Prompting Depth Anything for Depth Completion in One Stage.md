# Any to Full: Prompting Depth Anything for Depth Completion in One Stage

- **학회:** arXiv 2026, `arXiv:2603.05711` *(학회 게재 여부는 본문/공개 페이지 기준 미확인)*
- **링크:** https://arxiv.org/abs/2603.05711
- **코드:** https://github.com/zhiyuandaily/Any2Full
- **분야:** Depth Completion, Monocular Depth Estimation, RGB-D Perception, Robotic Perception

---

## 1. 요약

- **Any2Full**은 RGB 이미지와 sparse/incomplete depth를 입력으로 받아 dense metric depth map을 예측하는 **one-stage depth completion framework**이다.
- 기존 RGB-D fusion 기반 depth completion은 학습 RGB domain과 특정 depth sampling pattern에 강하게 의존하여, domain shift나 sensor pattern 변화에 취약하다.
- 최근 MDE 기반 방법들은 pretrained monocular depth estimation model의 geometric prior를 활용하지만, 대개 relative depth를 sparse metric depth와 명시적으로 정렬한 뒤 refinement하는 **two-stage 구조**를 사용한다.
- 이러한 two-stage 방식은 추가 계산 비용이 크고, MDE의 scale inconsistency 때문에 output-level alignment 과정에서 구조적 왜곡과 artifact가 생길 수 있다.
- Any2Full은 depth completion을 **pretrained MDE model의 scale-prompting adaptation**으로 재정의한다.
- Sparse depth는 dense depth를 직접 만들기 위한 입력이 아니라, MDE feature를 조절하기 위한 **scale cue**로 사용된다.
- 핵심 모듈은 **Scale-Aware Prompt Encoder, SAPE**이며, sparse하고 불규칙한 depth input을 MDE geometry의 안내를 받아 unified scale prompt로 변환한다.
- SAPE는 **Local Enrichment**, **Global Propagation**, **Scale Prompt Fusion**으로 구성된다.
- 최종적으로 MDE는 scale-consistent relative depth를 예측하고, 이를 sparse metric depth와 least-squares fitting으로 정렬해 dense metric depth를 얻는다.
- 논문은 Any2Full이 OMNI-DC 대비 평균 AbsREL을 32.2% 개선하고, 동일 MDE backbone 기준 PriorDA보다 1.4배 빠르다고 보고한다.

---

## 2. 핵심 기여

- **One-stage depth completion formulation 제안**
  - 중간 coarse metric depth 생성이나 별도 refinement network 없이, MDE feature 자체를 scale prompt로 조절한다.
  - 기존 two-stage MDE 활용 방식의 output-level alignment artifact와 refinement overhead를 줄인다.

- **Scale-Aware Prompt Encoder, SAPE 설계**
  - Sparse depth에서 scale cue를 추출하고, MDE의 dense geometric prior를 이용해 pattern-invariant한 scale prompt로 변환한다.
  - Local Enrichment로 patch 단위 scale cue를 MDE latent space에 anchor하고, Global Propagation으로 geometry-aware scale diffusion을 수행한다.

- **Domain-general & pattern-agnostic depth completion 달성**
  - RGB-D fusion 모델처럼 특정 RGB domain이나 depth pattern에 과적합되지 않도록 설계했다.
  - Hole, Range, Sparse-Random, Sparse-LiDAR, Sparse-SfM, Mixed pattern에서 강건성을 평가했다.

- **효율적인 real-world 적용 가능성 입증**
  - MDE backbone을 freeze하고 prompt encoder만 학습하는 lightweight 구조이다.
  - 실제 robotic warehouse black package grasping에서 성공률을 28%에서 91.6%로 개선했다고 보고한다.

---

## 3. 방법

### 입력

- **RGB image**  
  - \( I \in \mathbb{R}^{H \times W \times 3} \)

- **Sparse / incomplete metric depth**  
  - \( D_s \)
  - LiDAR, ToF, RGB-D sensor, SfM 등에서 얻은 희소하거나 결측이 있는 depth map

- **Pretrained MDE model**  
  - 논문에서는 **Depth Anything v2**를 backbone으로 사용
  - released relative-depth model로 초기화하고, MDE backbone은 freeze

---

### 핵심 아이디어

#### 1) Depth completion을 scale prompting 문제로 재정의

기존 방식은 RGB와 sparse depth를 직접 fusion하여 dense metric depth를 학습한다.

\[
\hat{D}_f = F_\theta(I, D_s)
\]

Any2Full은 이 대신 pretrained MDE model을 유지하고, sparse depth에서 얻은 scale cue를 prompt로 주입한다.

\[
\hat{\tilde{D}}_f = \mathcal{M}\left(I \mid \mathcal{G}(\tilde{D}_s)\right)
\]

- \(\mathcal{M}\): pretrained MDE model
- \(\mathcal{G}\): Scale-Aware Prompt Encoder, SAPE
- \(\tilde{D}_s\): normalized sparse depth
- \(\hat{\tilde{D}}_f\): scale-consistent relative depth prediction

여기서 MDE는 여전히 relative depth를 예측하지만, sparse depth에서 온 scale prompt 덕분에 global scale/bias로 metric depth에 잘 정렬 가능한 형태가 된다.

---

#### 2) Sparse depth normalization

Sparse metric depth \(D_s\)를 그대로 사용하지 않고, global scale과 bias를 제거한 \(\tilde{D}_s\)로 변환한다.

목적은 다음과 같다.

- 절대 scale과 bias의 직접 주입을 줄임
- sparse point 간 scale ratio 보존
- MDE의 relative depth representation과 더 잘 맞는 scale cue 생성

---

#### 3) Scale-Aware Prompt Encoder, SAPE

SAPE는 sparse하고 불규칙한 depth input을 MDE geometry의 안내를 받아 unified scale prompt로 바꾸는 핵심 모듈이다.

구성은 다음과 같다.

##### A. Local Enrichment

Sparse depth에서 얻은 patch-level depth feature \(F_{dep}\)와 MDE feature \(F_{mde}\)를 FiLM 방식으로 결합한다.

\[
f_{loc,i}
=
\gamma(f_{dep,i}, f_{mde,i}) \odot f_{mde,i}
+
\beta(f_{dep,i}, f_{mde,i})
\]

- \(f_{dep,i}\): i번째 patch의 depth feature
- \(f_{mde,i}\): i번째 patch의 MDE feature
- \(\gamma, \beta\): lightweight MLP가 예측
- 결과: \(F_{loc}\), local scale-aware feature

**의미:** sparse depth cue가 MDE feature를 대체하는 것이 아니라, MDE latent space 안에서 scale 관련 feature를 가볍게 조절한다.

---

##### B. Global Propagation

Local Enrichment 결과인 \(F_{loc}\)는 patch별 scale cue를 갖지만, 전역적으로 일관되지는 않다. Global Propagation은 MDE geometry를 따라 scale cue를 장면 전체로 확산한다.

초기값은 다음과 같다.

\[
F_{glo}^{0} = F_{loc}
\]

이후 geometry-guided Transformer block을 통해 업데이트한다.

\[
F_{glo}^{l}
=
\text{TransformerBlock}
\left(
F_{glo}^{l-1},
F_{mde}^{\phi(l)-1}
\right)
\]

중요한 설계는 attention weight를 sparse depth feature가 아니라 MDE feature에서 계산한다는 점이다.

```text
Query = MDE feature
Key   = MDE feature
Value = scale-aware feature
```

**의미:** scale cue가 sparse depth sampling pattern을 따라 퍼지는 것이 아니라, RGB 기반 MDE가 인식한 object boundary, surface, scene geometry를 따라 퍼진다.

---

##### C. Scale Prompt Fusion

Global Propagation으로 얻은 multi-level scale prompt \(\{F_{glo}^{1}, ..., F_{glo}^{L}\}\)를 MDE decoder에 FiLM 방식으로 주입한다.

\[
F_{mde}^{\prime \phi(l)}
=
\gamma^{l}(F_{glo}^{l}) F_{mde}^{\phi(l)}
+
\beta^{l}(F_{glo}^{l}, F_{mde}^{\phi(l)})
\]

- 각 decoder level에 대응하는 scale prompt를 주입
- token-wise modulation 수행
- MDE의 domain-general geometric prior는 유지하면서 scale consistency를 강화

---

#### 4) 최종 metric depth 변환

MDE decoder의 출력은 바로 metric depth가 아니라 scale-consistent relative depth이다.

\[
\hat{\tilde{D}}_f
\]

이를 valid sparse depth 위치에서 \(D_s\)와 맞추기 위해 non-parametric least-squares fitting을 수행한다.

개념적으로는 다음과 같다.

\[
\hat{D}_f = a \hat{\tilde{D}}_f + b
\]

- \(a\): global scale
- \(b\): global bias
- closed-form alignment이므로 추가 learnable module은 없음

---

#### 5) Training

Any2Full은 정확한 metric ground truth가 필요하기 때문에 high-quality synthetic RGB-D dataset으로 학습한다.

사용한 데이터:

- Hypersim indoor 60K
- VKITTI2 outdoor 10K
- TartanAir indoor/outdoor 15K

Depth sampling 전략:

- **Random Sampling:** 다양한 density의 sparse depth 생성
- **Hole Sampling:** 큰 연속 결측 영역을 갖는 depth input 생성

Loss 구성:

- \(L_{ssi}\): scale- and shift-invariant loss, global consistency용
- \(L_{gm}\): gradient matching loss, edge 보존용
- \(L_{anchor}\): valid sparse depth anchor에서의 consistency용
- \(L_{r\text{-}ssim}\): relative-structure SSIM loss, 구조적 유사성 정규화용

학습 설정:

- Backbone: Depth Anything v2
- MDE backbone: freeze
- Trainable module: Scale-Aware Prompt Encoder
- Steps: 224K
- Warm-up: 10K
- Scheduler: cosine
- Batch size: 16
- Optimizer: Adam
- Learning rate: \(5 \times 10^{-5}\)

---

### 출력

- **Dense metric depth map**
  - \(\hat{D}_f \in \mathbb{R}^{H \times W}\)

- 중간적으로 생성되는 출력:
  - normalized sparse depth \(\tilde{D}_s\)
  - scale prompt \(F_{glo}^{l}\)
  - scale-consistent relative depth \(\hat{\tilde{D}}_f\)

---

## 4. 메모

- Any2Full의 핵심은 sparse depth를 dense depth 생성용으로 직접 fusion하는 것이 아니라, **MDE가 scale-consistent하게 예측하도록 feature-level prompt를 주는 것**이다.

- 기존 PriorDA류 방식과의 가장 큰 차이는 다음과 같다.

```text
PriorDA류:
MDE relative depth 생성 → sparse depth와 output-level alignment → coarse metric depth → refinement

Any2Full:
sparse depth scale cue → MDE feature modulation → scale-consistent relative depth → global fitting
```

- 논문에서 말하는 one-stage는 final metric depth를 얻기 전까지 추가 refinement network나 intermediate depth prediction이 없다는 의미이다. 마지막 least-squares fitting은 closed-form alignment이므로 one-stage 정의에 포함된다.

- Global Propagation에서 Query/Key를 MDE feature로 사용하고 Value만 scale-aware feature로 사용하는 설계가 중요하다. 이 구조 덕분에 scale cue가 sparse depth의 불규칙한 sampling 위치가 아니라, MDE가 추정한 scene geometry를 따라 전파된다.

- Pattern-agnostic 성능의 근거:
  - sparse depth는 valid point에서 scale cue만 제공
  - local/global scale prompt 구성은 MDE geometry에 의존
  - Hole, Range, Sparse-Random, Sparse-LiDAR, Sparse-SfM, Mixed 등 다양한 pattern에서 평가

- Domain-general 성능의 근거:
  - Depth Anything v2 backbone을 활용
  - backbone을 freeze하여 pretrained MDE의 geometric prior를 보존
  - feature-level FiLM modulation으로 MDE representation을 최소한으로만 변경

- 실험적으로 Any2Full은 OMNI-DC 대비 평균 AbsREL을 32.2% 개선하고, 동일 MDE backbone 기준 PriorDA보다 1.4배 빠르다고 보고한다.

- 실제 로봇 물류창고 black package grasping 환경에서도 적용되었으며, grasping success rate를 28%에서 91.6%로 개선했다고 보고한다.

- 적용 관점에서 Any2Full은 다음 상황에 특히 유리하다.
  - ToF/RGB-D sensor의 hole이 많은 경우
  - LiDAR나 SfM처럼 sparse depth만 있는 경우
  - sensor range limitation으로 일부 depth range만 관측되는 경우
  - 학습 domain과 실제 deployment domain이 다른 경우

- 한계로 예상되는 부분:
  - 최종 metric depth는 여전히 sparse depth와의 global scale/bias fitting에 의존한다.
  - valid sparse depth가 극단적으로 적거나 특정 물체/거리 영역에 편향되어 있으면 scale fitting이 불안정할 수 있다.
  - MDE backbone의 geometry prior가 실패하는 장면에서는 scale prompt만으로 구조 오류를 완전히 보정하기 어렵다.

---

## 5. 적용 포인트
 - Sparse depth와 RGB를 분리해서 encoding함으로 scale이 multi-task 환경에 depth외 다른 task에 주는 영향을 줄일 수 있을 것 같음. 
