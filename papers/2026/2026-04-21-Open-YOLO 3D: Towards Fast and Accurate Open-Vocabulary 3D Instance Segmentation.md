# 제목: Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation

- **학회:** ICLR 2025 (Oral)
- **링크:** [arXiv](https://arxiv.org/html/2406.02548v3), [OpenReview](https://openreview.net/forum?id=CRmiX0v16e)
- **코드:** [OpenYOLO3D GitHub](https://github.com/aminebdj/OpenYOLO3D)
- **분야:** Open-vocabulary 3D instance segmentation, 3D vision, point cloud understanding

---

## 1. 요약
- 기존 open-vocabulary 3D instance segmentation은 SAM, CLIP, multi-view 3D feature aggregation에 크게 의존해서 추론 시간이 매우 느리다.
- 이 논문은 2D segmentation 기반 dense feature lifting 대신, **2D open-vocabulary detector의 bbox label**과 **3D instance proposal mask**를 결합하는 더 가벼운 구조를 제안한다.
- 핵심은 3D에서 먼저 class-agnostic instance mask를 만들고, 여러 view의 2D bbox 결과를 모아 각 3D mask의 class를 정하는 것이다.
- 이를 위해 LG(Label Map), VAcc(Accelerated Visibility Computation), MVPDist(Multi-View Prompt Distribution)를 도입한다.
- 저자들의 주장은, 3D instance의 projection 자체가 이미 instance 정보를 상당 부분 담고 있으므로 SAM 같은 2D segmentation refinement는 중복일 수 있다는 것이다.
- 결과적으로 ScanNet200, Replica에서 강한 성능을 내면서도 기존 방법 대비 최대 약 16배 빠른 속도를 달성했다.

---

## 2. 핵심 기여
- **2D object detection 기반 open-vocabulary 3D labeling**: 2D segmentation 대신 2D detector의 bbox와 label만 사용해 3D instance에 class를 부여한다.
- **Low-Granularity label map + Multi-View Prompt Distribution**: 여러 view의 coarse 2D class evidence를 3D mask에 집계해 robust하게 class를 추정한다.
- **가속된 visibility 계산**: point cloud projection과 mask visibility 계산을 tensor/GPU 연산으로 처리해 빠르게 top-k view를 선택한다.
- **속도-정확도 trade-off 개선**: Open3DIS 대비 ScanNet200 val에서 mAP50 기준 절대 2.3% 향상과 약 16배 속도 향상을 보고한다.

---

## 3. 방법

### 입력
- 3D reconstructed point cloud scene \(P\)
- 이 장면에 대응되는 multi-view RGB-D frame \((I, D)\)
- 각 frame의 intrinsic / extrinsic / pose 정보
- open-vocabulary 2D detector와 3D instance segmentation network의 pretrained model 출력

### 핵심 아이디어
- **3D proposal 생성**: Mask3D를 이용해 class-agnostic 3D instance mask proposal을 만든다. 3D에서 먼저 instance를 안정적으로 확보하는 단계다.
- **2D detection 수행**: 각 RGB frame에 대해 open-vocabulary 2D detector로 bbox와 class label을 예측한다.
- **LG label map 구성**: 각 frame에서 bbox 영역을 class id로 채운 coarse 2D label map을 만든다. pixel-wise segmentation이 아니라 bbox 단위의 저해상 semantic map이다.
- **VAcc로 visibility 계산**: 모든 3D point를 모든 frame으로 projection하고, frame 내부 여부와 depth 기반 occlusion 여부를 함께 계산해 각 3D mask가 frame별로 얼마나 잘 보이는지 측정한다.
- **Top-k view 선택**: 각 3D mask proposal마다 visibility가 높은 frame들만 선택한다.
- **MVPDist 생성**: 선택된 frame들에서 3D mask에 속한 point들이 떨어지는 LG label map의 class들을 모아 multi-view label distribution을 만든다.
- **최종 class 결정**: MVPDist에서 가장 많이 등장한 class를 해당 3D mask의 label로 할당한다.
- **confidence score**: class occurrence 기반 \(s_{class}\)와 projected 3D mask bbox와 2D detector bbox의 multi-view IoU 기반 \(s_{IoU}\)를 곱해 최종 score를 만든다.

### 출력
- 각 3D instance mask에 대해
  - class-agnostic 3D mask
  - open-vocabulary class label
  - confidence score  
  를 포함한 최종 3D instance segmentation 결과

---

## 4. 메모
- 이 논문은 **“3D는 instance, 2D는 semantic cue”**로 역할을 분리한 설계라고 보면 된다.
- dense 2D segmentation feature를 3D로 lift하는 기존 계열보다 구조가 단순하고 추론이 훨씬 빠르다.
- 반면 매우 작은 물체처럼 3D proposal network가 잘 놓치는 경우에는 한계가 있다. 저자들도 FastSAM 같은 빠른 2D instance segmentation을 이용한 3D proposal 보강이 향후 개선 방향이라고 적고 있다.
- 논문의 실전 포인트는 “SAM/CLIP 기반 정밀 2D-3D lifting이 항상 필요한 것은 아니며, bbox 기반 multi-view voting만으로도 충분히 강한 3D open-vocabulary labeling이 가능하다”는 점이다.

---

## 5. 적용 포인트
 - 3D bbox GT를 labeling 하는데 활용할 수 있을 것으로 생각됨. 
