# Open-Vocabulary Monocular 3D Detection

- **학회:** 미확인 / 제공 본문 내 명시 없음
- **링크:** 미확인 / 제공 본문 내 명시 없음
- **코드:** 공개 예정 / 제공 본문 기준
- **분야:** Open-Vocabulary Monocular 3D Object Detection, Monocular 3D Detection, 3D Vision, Vision Foundation Model

---

## 1. 요약
- 이 논문은 단일 RGB 이미지에서 임의의 카테고리 객체를 metric 3D 공간에서 검출하는 **Open-Vocabulary Monocular 3D Detection(OVMONO3D)** 과제를 제안한다.
- 기존 3D object detection은 LiDAR, multi-view setup, point cloud 입력에 의존하거나 closed vocabulary 환경에 머무르는 한계가 있다.
- OVMONO3D는 open-vocabulary 2D detection과 monocular 3D detection의 교차 영역으로, 학습 중 보지 못한 novel category까지 3D bounding box로 검출하는 것을 목표로 한다.
- 핵심 문제는 크게 두 가지이다. 첫째, 고품질 3D bounding box annotation이 부족하여 일반화 가능한 모델 학습이 어렵다. 둘째, 기존 3D dataset에는 missing annotation과 naming ambiguity가 있어 평가가 불안정하다.
- 이를 해결하기 위해 논문은 2D detection과 3D box prediction을 분리하는 decoupled framework를 제안한다.
- 첫 번째 방법인 **OVMONO3D-GEO**는 OV 2D detector, SAM, metric depth estimator를 사용해 2D detection 결과를 3D point cloud로 unprojection하고, PCA/DBSCAN 기반으로 3D box를 생성하는 training-free baseline이다.
- 두 번째 방법인 **OVMONO3D-LIFT**는 DINOv2 image feature, pseudo metric depth, point map ROI feature를 결합하여 2D box를 class-agnostic 3D cuboid로 lift하는 learning-based 방법이다.
- 평가 측면에서는 기존 AP3D/IoU3D metric의 한계를 보완하기 위해, 각 이미지에서 ground-truth로 존재하는 category만 prompt로 사용하는 **target-aware metric**을 제안한다.
- 실험은 Omni3D와 CityScapes3D에서 수행되며, OVMONO3D-LIFT는 novel category zero-shot 3D detection과 base category in-domain detection 모두에서 강한 성능을 보인다.
- 분석 결과, 이 과제의 주요 병목은 object depth prediction이며, DINOv2와 같은 3D-aware image feature, 정확한 metric depth estimation, 충분한 training data scale이 성능에 중요하다는 점을 확인한다.

---

## 2. 핵심 기여
- **새로운 과제 정의:** 단일 RGB 이미지에서 임의의 category 객체를 metric 3D 공간에서 검출하는 **Open-Vocabulary Monocular 3D Detection(OVMONO3D)** 문제를 제안하고 체계적으로 연구한다.
- **효과적인 decoupled framework 제안:** open-vocabulary 2D detector로 2D recognition/localization을 수행하고, 이를 class-agnostic 3D lifting head로 3D cuboid로 변환하는 **OVMONO3D-LIFT**를 제안한다. 또한 training-free geometric baseline인 **OVMONO3D-GEO**도 함께 제시한다.
- **평가 프로토콜 개선:** missing annotation과 semantic naming ambiguity 문제를 완화하기 위해, 이미지별 ground-truth category만 평가 prompt로 사용하는 **target-aware evaluation metric**을 제안한다.

---

## 3. 방법

### 입력
- 단일 RGB 이미지
- 카메라 intrinsic matrix
- Text prompt 또는 category list
- Open-vocabulary 2D detector가 예측한 2D bounding boxes
- 학습 시에는 base category에 대한 3D bounding box annotation
- 학습 및 추론 시 pseudo depth map은 pretrained metric depth estimator에서 획득하며, ground-truth depth는 필요하지 않음

### 핵심 아이디어
- 전체 문제를 한 번에 end-to-end로 해결하지 않고, 다음 두 단계로 분리한다.
  - **1단계:** Grounding DINO 또는 YOLO-World 같은 open-vocabulary 2D detector로 객체의 2D 위치와 category를 검출한다.
  - **2단계:** 검출된 2D box를 class-agnostic 3D lifting module에 넣어 3D bounding box로 변환한다.

- **OVMONO3D-GEO**
  - OV 2D detector로 2D box와 category를 얻는다.
  - SAM으로 instance mask를 추정한다.
  - UniDepthv2와 같은 metric depth estimator로 depth map을 예측한다.
  - mask 내부 pixel을 camera intrinsic으로 3D point cloud로 unprojection한다.
  - PCA로 object orientation을 추정하고, DBSCAN으로 outlier를 제거한다.
  - 정제된 point cloud에서 centroid, dimension, orientation을 계산해 3D bounding box를 생성한다.
  - 3D annotation 없이 동작하지만, depth 품질과 segmentation 품질에 민감하고 occlusion 상황에서 성능이 떨어진다.

- **OVMONO3D-LIFT**
  - DINOv2 encoder로 image feature를 추출하고 Feature Pyramid module로 multi-scale feature를 생성한다.
  - 2D box를 ROI로 사용하여 visual ROI feature를 얻는다.
  - metric depth estimator가 예측한 depth map을 camera intrinsic으로 point map으로 변환한다.
  - point map에서도 ROI pooling을 수행하여 local geometric feature를 얻는다.
  - visual feature와 geometric feature를 concat하여 geometry-informed feature를 구성한다.
  - class-agnostic 3D cube head가 depth, 3D size, 3D center, pose 등 3D box attribute를 예측한다.
  - Cube R-CNN과 달리 class-specific layer나 per-class average size prior에 의존하지 않기 때문에 novel category 일반화에 유리하다.

- **OVMONO3D-LIFT\***
  - OVMONO3D-LIFT에서 depth estimation module을 제거한 variant이다.
  - visual feature만으로 3D bounding box를 예측한다.
  - full LIFT와 비교하여 geometric depth information의 기여도를 분석하기 위한 baseline이다.

- **Target-aware metric**
  - 기존 AP3D/IoU3D 평가는 missing annotation과 naming ambiguity 때문에 open-vocabulary setting에서 불리하거나 부정확할 수 있다.
  - 이를 완화하기 위해 각 이미지에서 annotation에 존재하는 category만 2D detector에 prompt로 제공한다.
  - 한 이미지에서 특정 category가 annotation되어 있다면, 해당 category의 instance들은 비교적 완전하게 라벨링되었을 가능성이 높다는 가정을 사용한다.
  - 이를 통해 누락 라벨과 table/desk, chair/sofa, vase/potted plant 같은 semantic ambiguity의 영향을 줄인다.

### 출력
- 각 객체에 대한 2D bounding box
- 각 객체의 category label
- 각 객체의 metric 3D bounding box
  - 3D center
  - depth
  - width, height, length
  - 3D orientation / pose
- Novel category 및 base category에 대한 AP3D / target-aware AP3D 평가 결과

---

## 4. 메모
- 이 논문은 open-vocabulary 2D detection의 강력한 category generalization 능력과 monocular 3D detection의 metric localization 능력을 결합하려는 연구이다.
- 핵심 설계는 **2D recognition은 OV detector에 맡기고, 3D geometry 추정은 class-agnostic lifting head가 담당하도록 분리한 것**이다.
- OVMONO3D-GEO는 구현이 단순하고 3D annotation이 필요 없지만, occlusion, noisy depth, imperfect segmentation에 취약하다.
- OVMONO3D-LIFT는 DINOv2 feature와 metric depth 기반 point map feature를 함께 사용하여 GEO보다 더 robust하다.
- 분석 결과 object depth prediction이 가장 큰 bottleneck으로 나타난다. 즉, monocular 3D open-vocabulary detection에서 좋은 depth prior 또는 depth-aware representation이 매우 중요하다.
- DINOv2가 가장 좋은 pretrained feature extractor로 나타났으며, 이는 DINOv2 feature가 depth, multi-view correspondence, relative pose 같은 3D-aware 정보를 어느 정도 포함하기 때문으로 해석된다.
- Grounding DINO가 YOLO-World보다 OVMONO3D에 더 적합한 2D detector로 나타났다.
- Training data scaling law 분석에서 데이터 크기가 커질수록 AP3D가 향상되는 경향이 관찰되며, 이 과제는 여전히 dataset scale에 크게 의존한다.
- 기존 3D dataset은 missing label과 naming ambiguity가 심하기 때문에, open-vocabulary 3D detection에서는 평가 프로토콜 자체가 매우 중요하다.
- 한계로는 in-the-wild 이미지에서 정확한 camera intrinsic을 얻기 어렵다는 점, COCO와 같은 일반 이미지 데이터셋에는 3D ground truth가 없어 정량 평가가 어렵다는 점, Grounding DINO와 DINOv2 사용으로 inference가 무겁다는 점이 있다.

---

## 5. 적용 포인트
 - depth와 open-vocabulary detection을 fusion하는 mono 3D openvoca detection baseline 방법으로 참고할 만한 논문
 - Omni3D dataset 활용 가능
