# Rooms from Motion: Un-posed Indoor 3D Object Detection as Localization and Mapping

- **학회:** NeurIPS 2025 / arXiv:2505.23756
- **링크:** https://arxiv.org/abs/2505.23756
- **코드:** 공식 공개 코드 미확인
- **분야:** Object-centric 3D Object Detection, Localization & Mapping, SfM/SLAM, Indoor 3D Scene Understanding

---

## 1. 요약

- **Rooms from Motion(RfM)**은 포즈가 주어지지 않은 RGB 또는 RGB-D 이미지 컬렉션에서 **metric camera pose**와 **semantic 3D object map**을 동시에 추정하는 객체 중심 프레임워크이다.
- 기존 SfM/SLAM이 2D keypoint와 3D point를 기본 primitive로 사용하는 것과 달리, RfM은 **oriented 3D bounding box**를 전체 파이프라인의 기본 primitive로 사용한다.
- 기존 3D object detection 방법들은 보통 사전 camera pose, depth, point cloud, voxel volume에 의존하지만, RfM은 point cloud나 dense volume 없이도 객체 단위의 sparse representation만으로 localization과 mapping을 수행한다.
- 각 이미지에서 **Cubify Transformer(CuTR)**를 사용해 metric-scale 3D object box, object embedding, class label, score를 예측한다.
- 이미지 쌍 사이에서는 **Cubify Match**가 object-level matching과 box-corner-level matching을 수행하고, matched 3D box corner를 이용해 relative pose를 추정한다.
- 여러 이미지 쌍의 relative pose를 모아 view graph를 구성한 뒤, rotation averaging과 translation averaging으로 global camera pose를 계산한다.
- 이후 matched object들을 union-find로 연결해 **object track**을 만들고, 각 track을 global frame으로 올려 대표 3D box를 선택한다.
- 사전 pose가 있는 경우 또는 pose 추정 후에는, object track의 box corner reprojection error를 줄이는 partial bundle adjustment로 global 3D box 품질을 개선한다.
- 실험에서는 CA-1M과 ScanNet++에서 point-based 및 multi-view 3D object detection 방법보다 높은 map quality를 보였고, RGB-only / RGB-D 설정에서도 강한 metric localization 성능을 보였다.
- 한계로는 현재 indoor scene 중심이며, 객체가 거의 없는 이미지에서는 registration이 어렵고, 더 다양한 실내/실외 데이터와 카메라 조건이 필요하다는 점이 있다.

---

## 2. 핵심 기여

- **Object-centric SfM 구조 제안**
  - 기존 SfM의 `keypoint detection → keypoint matching → pose estimation → point track → bundle adjustment` 구조를 `3D object detection → object matching → box corner matching → pose estimation → object track → object-level optimization` 구조로 치환했다.
  - 즉, point cloud를 중간 표현으로 만들지 않고, 3D box만으로 localization과 mapping을 수행한다.

- **Un-posed 3D object detection 문제 설정**
  - 기존 multi-view 3D object detection은 보통 GT pose 또는 사전 pose를 요구한다.
  - RfM은 unordered, un-posed RGB/RGB-D image collection에서 camera pose와 global 3D object map을 동시에 추정하므로, 3D object detection을 localization-and-mapping 문제로 재정의한다.

- **Sparse하고 확장 가능한 object map 표현**
  - dense voxel, point cloud, radiance field처럼 장면의 공간 해상도에 비례하는 표현을 쓰지 않는다.
  - 장면 내 객체 수에 비례하는 oriented 3D box representation을 사용하므로, 작은 메모리 footprint로 semantic 3D map을 구성할 수 있다.

---

## 3. 방법

### 입력

- 순서가 없는 RGB 또는 RGB-D 이미지 컬렉션
- 사전 camera pose는 선택 사항
- depth map도 선택 사항
- 실험 설정:
  - GT depth + GT pose
  - GT depth only
  - monocular RGB + GT pose
  - monocular RGB only

---

### 핵심 아이디어

RfM은 3D oriented box를 SfM의 point primitive 대신 사용한다.

```text
기존 SfM:
2D keypoint
→ keypoint matching
→ relative pose
→ global pose
→ point track
→ bundle adjustment
→ sparse point map

Rooms from Motion:
3D object box
→ object matching
→ box corner matching
→ relative pose
→ global pose
→ object track
→ partial bundle adjustment
→ semantic 3D object map
```

#### 1) 이미지별 3D 객체 검출

각 이미지에 대해 Cubify Transformer를 실행한다.

```text
Image I
→ CuTR
→ (B, F, C, S)
```

- `B`: oriented 3D bounding boxes
- `F`: object feature embeddings
- `C`: classification labels
- `S`: classification scores

CuTR은 이미지 기반으로 metric-scale 3D box를 예측한다. 출력은 image resolution에 독립적이므로 고해상도 이미지에서 실행해 작거나 멀리 있는 객체 검출을 개선할 수 있다.

#### 2) Object-level matching

두 이미지 `I1`, `I2`에서 검출된 객체 집합을 matching한다.

```text
(B1, F1) ↔ (B2, F2)
```

- LightGlue 스타일의 self-attention / bidirectional-attention 구조를 object feature에 적용한다.
- 3D box의 center와 dimension에서 positional encoding을 만든다.
- 출력은 object pair별 matching score와 partial assignment이다.
- score threshold를 넘는 object pair만 matched object로 사용한다.

#### 3) Box/corner-level matching

Object-level matching만으로는 relative pose를 안정적으로 추정하기 어렵다. 객체는 partial observation이나 occlusion 때문에 box center나 임의의 corner가 실제 동일한 3D 지점에 대응한다고 보장하기 어렵기 때문이다.

따라서 각 3D box의 8개 corner를 모아 “box cloud”를 만들고, corner-level matching을 추가로 수행한다.

```text
각 object box → 8 corners
모든 box corner concat
→ N × 8 corner queries
→ corner-level matching
```

단, corner match는 object-level match와 일관되는 경우만 유지한다.

#### 4) Relative pose estimation

Matched 3D corners를 두 이미지 간 대응점으로 사용하여 Kabsch alignment로 relative pose를 추정한다.

```text
matched corners in I1
matched corners in I2
→ Kabsch alignment
→ relative pose R12
```

- 4-DoF alignment로 처리한다.
- 추정 대상은 yaw rotation과 translation이다.
- pitch/roll은 gravity measurement가 있다고 가정한다.
- CuTR의 3D box가 metric scale이므로 translation도 metric scale로 추정된다.

#### 5) Geometric verification

Object/corner matching에는 오류가 있을 수 있으므로 geometric verification을 수행한다.

- 최소 2개의 matched object pair로 relative pose 후보를 만든다.
- `I1`의 box를 추정된 pose로 `I2` frame에 재투영한다.
- 재투영된 box와 matched box 사이의 `IoU3D`를 계산한다.
- `IoU3D ≥ 0.25`이면 inlier로 간주한다.
- 전체 matched object 중 inlier 비율이 0.5 이상인 pose 후보만 유지한다.
- 평균 matching error가 가장 낮은 pose를 verified relative pose로 선택한다.

#### 6) View graph와 global pose estimation

모든 이미지 쌍의 verified relative pose로 view graph를 만든다.

```text
Node: image
Edge: verified relative pose
```

이후 glomap 스타일의 averaging을 수행한다.

- Rotation averaging 3회
  - 평균 결과와 3도 이상 불일치하는 relative rotation 제거
- Translation averaging 3회
  - metric scale이 필요하므로 scale parameter 고정
  - 평균 결과와 10cm 이상 불일치하는 relative translation 제거

최종적으로 largest connected component 안의 각 이미지에 대해 global pose `RT_i`를 얻는다.

#### 7) Object track establishment

Verified object matches에 대해 union-find를 수행하여 object track을 만든다.

```text
O_j = {(I_i, B_i)}
```

각 track은 여러 이미지에서 동일 객체로 관측된 3D box들의 집합이다.

각 `B_i`는 해당 이미지의 global pose `RT_i`를 이용해 global frame으로 변환된다. 이후 track 내 box들 중 다음 기준이 높은 box를 representative box로 선택한다.

- track 내 다른 box들과의 평균 mutual 3D IoU
- detection score

Semantic class가 있는 데이터셋에서는 observation별 classification score를 가중치로 class distribution을 만들고, 가장 확률이 높은 class를 대표 label로 사용한다.

#### 8) Track merging

초기 object track 생성 이후에도 같은 객체가 여러 track으로 분리될 수 있다.

RfM은 global pose가 있는 상태에서 track pair를 비교해 merging/suppression을 수행한다.

- generalized IoU 3D가 너무 낮은 track pair는 무시
- observation affinity = object matching score × shifted generalized IoU 3D
- track affinity가 0.25를 넘으면 merge
- merge 기준은 넘지 못하지만 IoU3D가 0.15보다 크면 suppress

#### 9) Partial bundle adjustment

Global 3D box는 각 이미지에서 partial observation 또는 occlusion으로 인해 부정확할 수 있다. 이를 개선하기 위해 object track 단위의 partial bundle adjustment를 수행한다.

핵심은 3D point를 독립적으로 최적화하지 않고, box parameter를 최적화하는 것이다.

```text
Optimized variables:
- box center
- box dimensions
- yaw

Cost:
- representative box corner를 각 observation에 projection
- expected 2D corner location과 reprojection error 최소화
```

즉, 전통적인 SfM의 point-track bundle adjustment를 box-corner-track optimization으로 바꾼 형태이다.

---

### 출력

- 각 이미지의 global metric camera pose
- 이미지 간 verified relative pose graph
- object track
- global oriented 3D bounding boxes
- semantic class label
- classification score
- 최종 semantic 3D object map

---

## 4. 메모

- RfM은 3D object detection을 단순한 detection 문제가 아니라 **localization + mapping 문제의 출력**으로 해석한다.
- 기존 point/voxel 기반 방법은 장면 전체의 geometry를 과매개변수화한 뒤 객체를 추출하는 반면, RfM은 처음부터 객체를 중심 표현으로 사용한다.
- 가장 중요한 가정은 CuTR이 단일 이미지에서 metric-scale 3D box를 충분히 잘 예측할 수 있다는 점이다.
- Localization 성능은 Cubify Match가 얼마나 많은 객체를 안정적으로 matching할 수 있는지에 크게 의존한다.
- Ablation 결과, 객체 taxonomy가 풍부할수록 registration rate와 pose 정확도가 좋아진다.
- 객체의 실제 class label 자체보다는 “매칭 가능한 객체 수”가 localization에 더 중요하다.
- Field of view가 넓을수록 한 이미지에 더 많은 객체가 보이고, frame 간 matching 가능성이 증가하므로 localization 성능이 좋아진다.
- RGB-only setting에서도 동작하지만, depth가 있으면 CuTR의 metric 3D box 예측과 pose estimation이 더 안정적이다.
- ScanNet++ DSLR처럼 sparse하고 smooth trajectory가 아닌 이미지 collection에서도 RfM은 전통적인 SLAM보다 강한 결과를 보인다.
- 한계는 객체가 없는 이미지에서는 register가 어렵고, 현재는 indoor scene 중심으로 학습 및 평가되었다는 점이다.
- 구현 관점에서 핵심 모듈은 `CuTR detector`, `object matcher`, `corner matcher`, `relative pose estimator`, `pose averaging`, `object track manager`, `box optimizer`로 나눌 수 있다.
- 실제 구현 시 가장 까다로운 부분은 box parameterization, 3D IoU 계산, corner correspondence filtering, global pose averaging, partial BA의 안정화일 가능성이 높다.

---

## 5. 적용 포인트
 - 객체가 적을수록 성능이 낮아질 수 있음. 
 - 3D ojbect 기반 slam
