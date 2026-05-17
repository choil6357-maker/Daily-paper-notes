# Training an Open-Vocabulary Monocular 3D Object Detection Model without 3D Data

- **학회:** NeurIPS 2024
- **링크:** https://ovm3d-det.github.io/
- **코드:** https://github.com/LeapLabTHU/OVM3D-Det
- **분야:** Open-Vocabulary 3D Object Detection, Monocular 3D Detection, RGB-only 3D Auto-labeling, Pseudo-LiDAR, Vision-Language Model

---

## 1. 요약

- 본 논문은 RGB 이미지만을 사용하여 open-vocabulary monocular 3D object detector를 학습하는 **OVM3D-Det** 프레임워크를 제안한다.
- 기존 open-vocabulary 3D detection 방법들은 주로 LiDAR나 RGB-D 센서 기반 point cloud를 필요로 하므로, 데이터 수집 및 배포 비용이 높다.
- OVM3D-Det은 고정밀 LiDAR나 3D 센서 없이, RGB 이미지에서 open-vocabulary 2D detector와 monocular depth estimator를 이용해 pseudo 3D bounding box를 자동 생성한다.
- 먼저 Grounded-SAM을 이용해 novel object의 2D bounding box와 mask를 얻고, UniDepth를 이용해 depth map을 예측한다.
- 이후 mask 내부 pixel을 camera intrinsic과 depth를 이용해 3D 공간으로 back-project하여 instance-level pseudo-LiDAR를 생성한다.
- 그러나 pseudo-LiDAR는 mask edge와 depth estimation error로 인해 noise가 많고, occlusion된 객체의 실제 크기 추정이 어렵다.
- 이를 해결하기 위해 객체 크기에 따라 erosion 강도를 조절하는 **adaptive pseudo-LiDAR erosion**을 제안한다.
- 또한 GPT-4와 같은 LLM에서 얻은 category-level size prior를 활용하여 비합리적인 3D box를 판별하고, bounding box search를 통해 box를 보정한다.
- 최종적으로 자동 생성된 pseudo 3D label을 사용해 Cube R-CNN 기반 monocular 3D detector를 학습하고, classification branch는 text embedding alignment 방식으로 수정한다.
- KITTI, nuScenes, SUN RGB-D, ARKitScenes에서 실험한 결과, OVM3D-Det은 novel category에서 baseline보다 크게 우수한 성능을 보였다.

---

## 2. 핵심 기여

- **RGB-only open-vocabulary monocular 3D detection 프레임워크 제안**
  - LiDAR point cloud나 3D sensor data 없이 RGB 이미지만으로 open-vocabulary 3D detector를 학습하는 최초의 image-based 3D open-vocabulary detection 프레임워크를 제안한다.
  - 학습 데이터 생성 단계에서도 LiDAR를 사용하지 않기 때문에, 인터넷 규모의 RGB 이미지 데이터로 확장할 수 있는 가능성이 있다.

- **Pseudo-LiDAR 기반 자동 3D pseudo label 생성**
  - Grounded-SAM으로 novel object를 2D detection / segmentation하고, UniDepth로 depth를 추정한 뒤, object mask 내부 pixel을 3D로 back-project하여 pseudo-LiDAR를 만든다.
  - 이를 통해 각 객체 instance에 대한 pseudo 3D point cloud와 3D bounding box를 자동 생성한다.

- **Noisy pseudo-LiDAR와 occlusion 문제를 해결하는 box refinement 설계**
  - Mask boundary에서 발생하는 pseudo-LiDAR noise를 줄이기 위해 adaptive erosion을 적용한다.
  - Occlusion이나 depth noise로 인해 box 크기가 과소/과대 추정되는 문제를 해결하기 위해 LLM 기반 category size prior와 bounding box search를 사용한다.
  - Ray tracing loss와 point ratio loss를 결합하여 가장 적절한 proposal box를 선택한다.

---

## 3. 방법

### 입력

- RGB image
- 관심 novel object category의 text query 또는 class list
- Camera intrinsic
  - `f_x, f_y, c_x, c_y`
- Off-the-shelf open-vocabulary 2D model
  - Grounding DINO + SAM 구조의 Grounded-SAM
- Pre-trained monocular depth estimator
  - UniDepth
- LLM 기반 category size prior
  - 예: GPT-4로부터 얻은 class별 typical length, width, height

---

### 핵심 아이디어

#### 1. Open-vocabulary 2D detection / segmentation

- RGB 이미지에 대해 Grounded-SAM을 사용한다.
- 관심 category text list를 query로 입력하면, 각 객체에 대해 다음을 얻는다.

```text
2D bounding box B_k
instance mask M_k
pseudo class label y_k
```

- 이 단계는 open-vocabulary 2D model의 zero-shot recognition 능력을 활용한다.

#### 2. Monocular depth estimation과 pseudo-LiDAR 생성

- UniDepth를 사용해 입력 이미지의 depth map `D(u, v)`를 예측한다.
- 각 pixel `(u, v)`에 대해 camera coordinate system에서의 3D 좌표를 계산한다.

```text
z = D(u, v)
x = (u - c_x) * z / f_x
y = (v - c_y) * z / f_y
```

- 각 object mask 내부 pixel을 위 식으로 back-project하여 instance-level pseudo-LiDAR point cloud를 생성한다.

```text
M_i 내부 pixel
→ depth 기반 back-projection
→ object-level pseudo-LiDAR point cloud
```

#### 3. Adaptive pseudo-LiDAR erosion

- Raw pseudo-LiDAR는 object boundary에서 noise가 크다.
- 원인은 foreground object와 background가 이미지에서는 인접해 있지만 실제 depth 차이는 클 수 있기 때문이다.
- 따라서 mask edge를 erosion하여 boundary noise를 제거한다.
- 단, 고정 kernel erosion은 작은 객체를 과도하게 제거하거나 큰 객체의 noise를 충분히 제거하지 못할 수 있다.
- 이를 해결하기 위해 mask 크기에 따라 erosion 강도를 조절한다.

```text
small object → weak erosion
large object → strong erosion
```

- 결과적으로 object 내부의 유효 point는 보존하면서 boundary artifact를 줄인다.

#### 4. Ground plane 추정과 좌표계 정렬

- 3D bounding box가 ground plane과 평행하다고 가정한다.
- Outdoor에서는 “ground”, indoor에서는 “floor” prompt를 Grounded-SAM에 입력하여 ground/floor mask를 얻는다.
- 해당 mask를 depth와 함께 3D로 back-project하고 least squares로 ground plane을 fitting한다.
- 이후 pseudo-LiDAR를 ground-parallel coordinate system으로 변환한다.

#### 5. Box orientation estimation

- 정제된 pseudo-LiDAR를 ground plane에 projection한다.
- 객체의 yaw orientation을 추정하기 위해 두 방법을 비교한다.
  - pseudo-LiDAR 내 point pair direction histogram
  - PCA 기반 principal direction 추정
- 실험적으로 PCA가 더 단순하고 효율적이며 성능도 좋았다.

#### 6. Coarse 3D bounding box 생성

- 추정된 orientation에 맞춰 refined pseudo-LiDAR를 tight하게 감싸는 3D bounding box를 만든다.
- 이 box를 coarse pseudo box로 사용한다.
- 하지만 occlusion이 있으면 object point가 일부만 관측되어 box가 작아질 수 있고, noise가 남아 있으면 box가 커질 수 있다.

#### 7. LLM size prior 기반 box reasonability check

- GPT-4와 같은 LLM에 class별 일반적인 실제 크기를 질문한다.

```text
Please provide the (length, width, height) for objects of the <CLASS> category according to their typical sizes in real life.
```

- LLM이 제공한 class별 size를 category prior로 사용한다.
- Coarse box의 dimension이 prior의 일정 범위 안에 있으면 valid pseudo label로 간주한다.

```text
valid range = [τ1 * prior, τ2 * prior]
```

- 너무 작거나 큰 box는 bounding box search를 통해 보정한다.

#### 8. Bounding box search

- Coarse pseudo box의 네 corner를 기준으로 proposal box를 생성한다.
- 각 corner마다 class dimension prior를 적용하여 두 방향의 box를 만들기 때문에, coarse box 하나당 총 8개의 proposal box가 생성된다.
- Proposal box 중 가장 적절한 box를 선택하기 위해 두 loss를 사용한다.

```text
L_search = L_trace + λ * L_point_ratio
```

- `L_trace`
  - Ray tracing loss
  - 각 pseudo-LiDAR point와, 해당 camera ray가 proposal box와 만나는 intersection point 사이의 거리를 계산한다.

- `L_point_ratio`
  - Proposal box 내부에 포함되는 object point 비율을 고려한다.
  - Box가 object point를 충분히 포함하지 못하는 shortcut 문제를 방지한다.

```text
L_point_ratio = 1 - N_inside / N_all
```

- 최종적으로 `L_search`가 가장 작은 proposal box를 training pseudo label로 선택한다.

#### 9. Cube R-CNN 기반 detector 학습

- 기본 monocular 3D detector로 Cube R-CNN을 사용한다.
- 기존 classification branch를 text-alignment head로 교체한다.
- Object feature와 text embedding의 dot product를 계산하여 class alignment를 수행한다.

```text
c_i = f_i · t
L_aligning = CE(c_i, y_i)
```

- 전체 학습 loss는 localization loss와 text alignment loss의 합이다.

```text
L_train = L_localization + L_aligning
```

---

### 출력

- 자동 생성된 pseudo 3D annotation
  - class label
  - 2D bounding box
  - instance mask
  - pseudo-LiDAR point cloud
  - refined 3D bounding box
  - yaw orientation
  - object dimension
- 학습된 open-vocabulary monocular 3D detector
  - 입력: single RGB image
  - 출력: open-vocabulary class에 대한 3D bounding boxes
- Novel category에 대한 3D detection 결과
  - KITTI
  - nuScenes
  - SUN RGB-D
  - ARKitScenes

---

## 4. 메모

- 이 논문의 핵심은 **3D sensor 없이 RGB-only 데이터로 open-vocabulary 3D detector를 학습할 수 있는가?**라는 질문에 대한 실용적 baseline을 제시한 것이다.
- OVM3D-Det은 inference뿐 아니라 training pseudo label 생성 단계에서도 LiDAR를 사용하지 않는다는 점이 중요하다.
- 기존 monocular 3D detector는 inference는 RGB만 사용하더라도 training에는 LiDAR 또는 3D annotation이 필요한 경우가 많았다.
- 이 방법은 Grounded-SAM과 UniDepth 같은 foundation model을 이용해 자동 pseudo-labeling pipeline을 구성한다.
- Pseudo-LiDAR는 real LiDAR보다 dense하지만 depth estimation error와 mask boundary artifact로 인해 noise가 크다.
- Adaptive erosion은 단순하지만 pseudo-LiDAR 품질 개선에 효과적인 구성 요소이다.
- LLM size prior는 occlusion으로 인해 객체 크기가 과소 추정되는 문제를 보정하는 데 사용된다.
- 논문에서는 GPT-4 prior와 dataset statistics 기반 prior가 유사한 성능을 보인다고 보고한다.
- 이는 LLM의 commonsense object size knowledge가 pseudo 3D label refinement에 유용할 수 있음을 보여준다.
- Box search에서 ray tracing loss만 사용하면 shortcut 문제가 발생할 수 있으므로 point ratio loss를 추가한다.
- Ablation study에서 naive framework 대비 전체 refinement를 적용했을 때 AP가 크게 향상되었다.
- KITTI, nuScenes 같은 outdoor dataset과 SUN RGB-D, ARKitScenes 같은 indoor dataset 모두에서 평가했다.
- Novel category 기준으로 baseline 대비 KITTI +6.7 AP, nuScenes +9.7 AP, SUN RGB-D +8.5 AP, ARKitScenes +16.8 AP 향상을 보였다.
- Point cloud 기반 open-vocabulary detector인 OV-3DET을 pseudo-LiDAR에 직접 적용하면 성능이 낮았으며, 이는 real point cloud와 pseudo-LiDAR 사이의 distribution shift 때문으로 해석된다.
- 최종 3D pseudo GT는 UniDepth 기반 depth와 camera intrinsic을 이용해 생성되므로 기본적으로 metric-scale pseudo label에 가깝다.
- 다만 실제 LiDAR GT가 아니라 depth estimator와 2D segmentation으로부터 생성된 pseudo GT이므로, scale error와 box error가 포함될 수 있다.
- 실내 로봇 환경에 적용하려면 floor plane, camera height, class size prior를 함께 사용해 pseudo 3D box의 안정성을 높이는 방향이 유용하다.
- 향후 확장 방향으로는 metric GT 대신 relative/canonical scale pseudo GT를 함께 학습하거나, metric scale recovery를 별도 모듈로 분리하는 hybrid 구조를 고려할 수 있다.

---

## 5. 적용 포인트
 - 3D bbox pseudo annotation 방법으로 활용가능
