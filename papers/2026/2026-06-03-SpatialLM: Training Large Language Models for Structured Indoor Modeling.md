# SpatialLM: Training Large Language Models for Structured Indoor Modeling

- **학회:** NeurIPS 2025
- **링크:** [arXiv](https://arxiv.org/abs/2506.07491) / [Project Page](https://manycore-research.github.io/SpatialLM/)
- **코드:** [GitHub](https://github.com/manycore-research/SpatialLM) / [Hugging Face Model](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B)
- **분야:** 3D Scene Understanding, Structured Indoor Modeling, Point Cloud LLM, Multimodal LLM, 3D Layout Estimation, 3D Object Detection

---

## 1. 요약

- **SpatialLM**은 RGB-D scan, LiDAR, 또는 monocular video reconstruction 등에서 얻은 **3D point cloud**를 입력으로 받아, 실내 장면의 구조를 **텍스트 기반 structured scene script**로 생성하는 3D Large Language Model이다.
- 출력은 벽, 문, 창과 같은 **architectural layout element**와 semantic category가 포함된 **oriented 3D object bounding box**로 구성된다.
- 기존 3D layout/object detection 방법처럼 task-specific decoder나 detection head를 설계하는 대신, 표준적인 **Encoder–MLP–LLM** 구조를 따르고 오픈소스 LLM을 직접 fine-tuning한다.
- 핵심 모델 구성은 **Point Cloud Encoder(Sonata / PTv3 계열) → MLP Projector → LLM(Qwen2.5-0.5B)** 이다.
- 자체 구축한 대규모 synthetic indoor dataset을 먼저 학습한 뒤, Structured3D 또는 ScanNet 같은 downstream benchmark에 fine-tuning하는 방식이 가장 좋은 성능을 보인다.
- 공개 benchmark에서 layout estimation은 SOTA 수준을 달성하고, 3D object detection에서는 전문 모델인 V-DETR과 경쟁 가능한 성능을 보인다.
- MASt3R-SLAM으로 RGB video에서 복원한 noisy point cloud에도 zero-shot으로 적용 가능함을 보여준다.

---

## 2. 핵심 기여

- **LLM 기반 structured indoor modeling 제안**
  - Point cloud로부터 실내 구조를 직접 예측하는 문제를 LLM의 **auto-regressive text generation** 문제로 변환한다.
  - 3D layout과 object box를 사람이 읽고 수정할 수 있는 Python-style script 형태로 표현한다.

- **대규모 고품질 synthetic indoor dataset 구축**
  - 전문 인테리어 디자인 repository를 활용하여 **12,328개 indoor scene / 54,778개 room** 규모의 데이터셋을 구축한다.
  - 데이터셋은 point cloud와 ground-truth 3D annotation을 포함한다.
  - 객체 annotation은 wall/door/window를 제외한 59개 일반 객체 category를 대상으로 하며, 최종적으로 **412,932개 object instance / 35,426개 unique CAD model**을 포함한다.

- **Point cloud encoder와 LLM alignment 전략 분석**
  - Mapping-based encoder, 3DCNN encoder, Sonata encoder를 비교한다.
  - 단순 voxelization이나 random sampling은 spatial information 손실로 성능이 낮다.
  - 학습 가능한 3D encoder가 효과적이며, 최종적으로 Sonata encoder가 가장 좋은 성능을 보인다.
  - 학습 schedule은 multi-stage보다 **single-stage fine-tuning**, 특히 encoder/projector/LLM 전체를 trainable로 두는 방식이 가장 효과적이다.

---

## 3. 방법

### 입력

- 입력은 일반적으로 RGB-D scan으로부터 얻은 3D point cloud이다.

```text
P ∈ R^{N × 6}
```

- 각 point는 다음 6차원 정보를 가진다.

```text
XYZ + RGB
```

- 논문에서는 RGB-D scan뿐 아니라 MASt3R-SLAM 등으로 RGB video에서 복원한 point cloud도 zero-shot 실험에 사용한다.

### 핵심 아이디어

- 실내 장면의 구조를 직접 box/layout tensor로 예측하지 않고, **general-purpose language script**로 표현한다.
- 전체 pipeline은 다음과 같다.

```text
Point Cloud P
  ↓
Point Cloud Encoder E
  ↓
K개의 visual token / point feature 생성
  ↓
MLP Projector
  ↓
LLM embedding space로 정렬
  ↓
LLM auto-regressive decoding
  ↓
Python-style scene script 생성
  ↓
script parsing
  ↓
3D layout + 3D object boxes 복원
```

- Point cloud encoder는 다음 mapping으로 정의된다.

```text
F = E(P),  P ∈ R^{N×6},  F ∈ R^{K×D},  K ≪ N
```

- 여기서 `K`는 LLM에 입력되는 visual token 수이며, point cloud의 공간 해상도와 LLM token length 사이의 trade-off를 결정한다.
- 최종 설정에서는 Sonata encoder를 사용하고, 가장 세밀한 spatial resolution을 **2.5cm**로 설정한다.
- 학습은 여러 stage로 나누지 않고, point cloud encoder, MLP projector, LLM을 한 번에 학습하는 **single-stage fine-tuning**이 가장 좋은 결과를 보인다.

### 출력

- 출력은 구조화된 scene description script이다.
- 포함되는 요소는 다음과 같다.

```text
1. Architectural layout
   - wall
   - door
   - window

2. 3D object detection result
   - semantic category
   - oriented 3D bounding box
```

- 최종적으로 script를 parsing하여 실내 장면의 구조 요소와 객체를 3D 공간상에 복원한다.

---

## 4. 메모

- **SceneScript와의 차이**
  - SceneScript도 structured indoor reconstruction을 sequence modeling으로 접근한다.
  - 하지만 SceneScript는 domain-specific token과 specialized Transformer decoder를 사용한다.
  - SpatialLM은 표준 MLLM 구조인 Encoder–MLP–LLM을 유지하고, 오픈소스 LLM을 직접 fine-tuning한다는 점이 다르다.

- **RoomFormer와의 차이**
  - RoomFormer는 2D density map에서 room polygon/corner를 예측한 뒤 3D로 extrusion하는 layout-specialist model이다.
  - SpatialLM은 point cloud를 직접 visual token으로 변환하고, LLM이 layout script를 생성한다.
  - Auto-regressive 생성 방식이기 때문에 wall-door-window 간 관계를 더 자연스럽게 유지할 수 있다.

- **V-DETR과의 차이**
  - V-DETR은 3D object detection 전용 DETR 계열 모델이며, 3DV-RPE와 object-based normalization 같은 task-specific 설계를 사용한다.
  - SpatialLM은 3D detection head를 따로 설계하기보다, object box를 language output으로 생성한다.

- **성능 경향**
  - Structured3D나 ScanNet만으로 LLM을 학습하면 성능이 낮다.
  - 대규모 자체 synthetic dataset으로 먼저 pre-training한 뒤 downstream dataset으로 fine-tuning하는 방식이 중요하다.
  - Layout estimation에서는 RoomFormer와 SceneScript를 능가한다.
  - 3D object detection에서는 SceneScript보다 우수하고, V-DETR과 경쟁 가능한 수준이다.

- **한계**
  - 임의의 모든 point cloud source에 대해 universal SOTA 모델은 아니다.
  - RGB-D, LiDAR, monocular video reconstruction 간 point cloud distribution 차이가 크기 때문에, 최고 성능을 위해서는 target dataset fine-tuning이 필요하다.
  - 현재는 predefined object category 기반이므로 open-vocabulary 3D detection은 제한적이다.
  - LLM의 기존 자연어 처리/추론 능력이 structured indoor modeling fine-tuning 후 얼마나 유지되는지는 충분히 평가되지 않았다.

- **실제 활용 관점**
  - RGB-D 또는 SLAM으로 얻은 실내 point cloud를 layout/object-level structured representation으로 변환하는 데 유용하다.
  - AR, embodied robotics, 실내 scene editing, navigation map generation 등에 활용 가능성이 있다.
  - 사용자가 자체 실내 데이터셋을 보유하고 있다면, 공개 코드와 checkpoint를 기반으로 fine-tuning하여 적용하는 방향이 현실적이다.

---

## 5. 적용 포인트
 - SpatialLM-Dataset 활용 가능
