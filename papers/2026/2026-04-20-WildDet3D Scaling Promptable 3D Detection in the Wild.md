# WildDet3D Scaling Promptable 3D Detection in the Wild

- **학회:** arxiv 2026
- **링크:** https://arxiv.org/pdf/2604.08626
- **코드:** https://github.com/allenai/WildDet3D
- **분야:** Open-vocabulary Monocular 3D Object Detection / 3D Perception

---

## 1. 요약
- WildDet3D는 **단일 RGB 이미지**로부터 물체의 **3D 위치, 크기, 방향**을 추정하는 **open-vocabulary monocular 3D detector**이다.
- 기존 방법들이 **text prompt만 지원**하거나 **box prompt만 지원**하는 반면, 이 논문은 **text / point / box / exemplar prompt**를 하나의 구조에서 통합한다.
- 또한 카메라 intrinsic이나 depth가 있으면 이를 활용하고, 없으면 내부 모듈로 추정하는 **geometry-aware** 구조를 제안한다.
- 핵심 구조는 **dual-vision encoder + promptable detector + 3D detection head**로 이루어진다.
- image encoder는 의미적 특징을, RGBD encoder는 깊이/기하 특징을 추출하고, 이를 fusion하여 3D 검출에 사용한다.
- 3D head는 depth, 2D spatial, semantic feature를 함께 사용해 3D bounding box를 예측한다.
- 학습 시에는 3D box regression뿐 아니라 **2D detection, depth estimation, camera geometry, confidence prediction**까지 함께 학습하는 **multi-task learning**을 사용한다.
- 데이터 측면에서는 2D detection dataset들로부터 후보 3D box를 생성하고, rule-based filtering과 human/VLM selection을 거쳐 **WildDet3D-Data**를 구축했다.
- 이 데이터셋은 **100만 장 이상 이미지, 13.5K 카테고리, 370만 개 유효 3D annotation**으로 구성되어 기존 Omni3D보다 훨씬 넓은 vocabulary를 제공한다.
- 결과적으로 WildDet3D는 WildDet3D-Bench, Omni3D, zero-shot transfer, real-depth setting에서 모두 강한 성능을 보였다.

---

## 2. 핵심 기여
- **통합형 open-vocabulary 3D detector 제안**: text, point, box, exemplar prompt를 모두 지원하고, optional depth까지 활용 가능한 unified monocular 3D detection architecture를 제안했다.
- **대규모 in-the-wild 3D detection dataset 구축**: 2D annotation 기반 후보 생성 → rule-based filtering → human/VLM selection 파이프라인으로 **WildDet3D-Data**를 만들었다.
- **강한 일반화 성능 입증**: WildDet3D-Bench, Omni3D, Argoverse 2, ScanNet, Stereo4D에서 SOTA 수준 결과를 보였고, 특히 depth가 주어질 때 큰 성능 향상을 확인했다.

---

## 3. 방법

### 입력
- 단일 RGB 이미지 `I`
- optional camera intrinsics `K`
- optional partial/full depth map `D`
- user prompt `P`
  - text prompt
  - point prompt
  - box prompt
  - exemplar prompt

### 핵심 아이디어
- **Dual-vision encoder**
  - image encoder는 semantic-rich dense feature를 추출
  - RGBD encoder는 optional depth를 포함한 geometry-aware feature를 추출
  - depth fusion module이 두 feature를 결합해 semantic + metric depth 정보를 함께 가진 representation 생성
- **Promptable detector**
  - text / point / box / exemplar를 하나의 prompt encoding framework로 통합
  - prompt token을 image feature와 cross-attention으로 결합해 detection query 생성
- **3D detection head**
  - query feature에 camera ray 정보와 depth latent를 cross-attention으로 추가
  - 3D center offset, log depth, log dimensions, rotation(6D representation), confidence를 예측
  - 회전/크기 중복 표현 문제를 줄이기 위해 **unambiguous rotation normalization** 적용
- **Multi-task learning**
  - 3D regression loss
  - 3D confidence loss
  - auxiliary geometry loss(depth / intrinsic / ray direction)
  - auxiliary 2D detection loss
  - one-to-many matching과 deep supervision으로 학습 안정성 및 수렴 속도 향상
- **Ignore-aware training**
  - 3D GT가 없는 visible object는 ignore 처리하고, evaluation/training에서 일관되게 false positive로 과도하게 벌주지 않도록 설계

### 출력
- scene 내 target object들에 대한 3D bounding boxes `{B_i}`
- 각 박스는 다음 정보를 포함
  - 3D center `c_i`
  - physical dimensions `(w, h, l)`
  - orientation `R_i`
  - confidence score `s_i`

---

## 4. 메모
- 이 논문의 **open-vocabulary 성격**은 class softmax를 고정해 두는 방식이 아니라, **text prompt 기반 조건부 검출**로 3D box를 예측한다는 점에서 나온다.
- 즉 class를 전체 vocabulary에서 직접 분류하기보다, **“이 prompt에 해당하는 물체를 찾아라”** 방식으로 동작한다.
- box prompt가 text prompt보다 성능이 더 높은 경우가 많았는데, 이는 **2D detection이 전체 성능의 병목**임을 보여준다.
- GT depth 또는 sparse depth가 들어가면 성능이 크게 올라가므로, 이 모델의 핵심 병목은 여전히 **monocular depth ambiguity**라고 볼 수 있다.
- 2D head를 제거하면 성능이 크게 무너지므로, 이 모델은 **2D detection prior 위에서 3D를 회귀하는 구조**가 매우 중요하다.
- WildDet3D-Data는 대규모 long-tail category를 제공하지만, 논문도 희귀 카테고리 성능의 불안정성과 long-tail 한계를 인정한다.
- rotation estimation은 여전히 어려운 문제로 남아 있으며, 대칭 물체나 시각 정보가 적은 경우 특히 불안정하다.
- 실제 응용(iPhone, AR, robotics, VLM integration)까지 보여주지만, 논문 스스로 **research prototype** 수준이며 safety-critical 용도에는 적합하지 않다고 명시한다.

---

## 5. 적용 포인트
 - 3D dataset을 생성하는 annotation 방법, Camera metrics를 spherical harmonic basis로 수학적 변환 후 MLP를 거쳐 cross attention하는 방법
