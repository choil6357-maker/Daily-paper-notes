# Distilling Monocular Foundation Model for Fine-grained Depth Completion

- **학회:** CVPR 2025
- **링크:** https://arxiv.org/pdf/2503.16970
- **코드:** https://github.com/Sharpiless/DMD3C
- **분야:** Depth Completion / Monocular Depth Estimation / Knowledge Distillation / 3D Vision

---

## 1. 요약
- 이 논문은 sparse LiDAR depth와 RGB를 이용하는 depth completion에서, sparse ground truth만으로는 세밀한 기하 정보를 학습하기 어렵다는 문제를 다룬다.
- 이를 해결하기 위해 monocular depth foundation model의 지식을 depth completion 모델로 옮기는 **2-stage distillation framework**를 제안한다.
- 1단계에서는 대규모 자연 이미지에 monocular depth model을 적용해 dense pseudo depth를 만들고, 이를 바탕으로 mesh reconstruction과 ray simulation으로 LiDAR-like sparse depth를 합성하여 pre-training 데이터를 생성한다.
- 이를 통해 실제 GT depth 없이도 다양한 장면에서 **상대적 깊이 관계와 기하 구조**를 미리 학습할 수 있게 한다.
- 2단계에서는 실제 sparse GT가 있는 데이터셋으로 fine-tuning을 수행하며, sparse supervision(L1 loss)과 monocular depth 기반 dense supervision을 함께 사용한다.
- 이때 monocular depth의 **scale ambiguity** 문제를 해결하기 위해 **SSI Loss (scale- and shift-invariant loss)** 를 도입한다.
- SSI Loss는 예측 depth와 monocular depth 사이의 최적 scale/shift를 맞춘 뒤 비교하므로, 절대 스케일 차이에 덜 민감하게 상대 구조를 학습할 수 있다.
- 추가로 gradient regularization을 사용해 depth discontinuity와 sharpness를 유지한다.
- 제안 방법은 KITTI depth completion benchmark에서 SOTA를 달성했고, NYU Depth V2에서도 우수한 성능을 보였다.
- 핵심 메시지는 **sparse supervision의 한계를 monocular foundation model의 dense geometric prior로 보완하자**는 것이다.

---

## 2. 핵심 기여
- monocular depth foundation model의 지식을 depth completion 모델로 전달하는 **2-stage distillation framework**를 제안했다.
- GT depth 없이도 활용 가능한 **synthetic data generation + pre-training 전략**을 제안했다.
- monocular depth의 scale ambiguity를 해결하기 위해 **SSI Loss**를 도입했다.
- sparse GT와 dense monocular supervision을 결합해, 세밀한 depth 구조와 실제 스케일 학습을 동시에 가능하게 했다.
- BP-Net, LRRU, CFormer 등 다양한 backbone에 적용 가능함을 보여 방법의 범용성을 입증했다.
- KITTI leaderboard 1위 및 zero-shot generalization 결과를 통해 성능과 일반화 능력을 검증했다.

---

## 3. 방법

### 입력
- RGB 이미지 \(I\)
- sparse depth map \(D_s\) (LiDAR 기반)
- 1단계 pre-training에서는 natural image와 monocular model이 생성한 dense pseudo depth
- 2단계 fine-tuning에서는 sparse GT depth와 monocular model이 예측한 dense depth

### 핵심 아이디어
- **1단계: 데이터 생성 및 사전학습**
  - 자연 이미지에 monocular depth model(예: Depth Anything V2)을 적용해 dense pseudo depth를 생성한다.
  - 임의의 camera intrinsic을 샘플링해 pseudo depth를 3D point cloud로 변환하고, mesh를 복원한다.
  - ray simulation으로 LiDAR scanning을 흉내 내 sparse depth를 생성한다.
  - 이렇게 만든 `(RGB, simulated sparse depth, pseudo dense depth)` 쌍으로 depth completion 모델을 pre-train한다.
  - 목적은 다양한 장면에서 dense geometric prior를 학습하는 것이다.

- **2단계: 실제 데이터셋 fine-tuning**
  - sparse GT가 있는 데이터셋에서 supervised L1 loss로 real-world scale을 맞춘다.
  - monocular dense depth를 추가 supervision으로 사용하되, 절대 scale mismatch를 직접 강제하지 않도록 **SSI Loss**를 사용한다.
  - SSI Loss는 예측 depth와 monocular depth 사이의 최적 scale/shift를 찾아 정렬한 뒤 loss를 계산한다.
  - 추가 gradient matching term으로 경계와 구조적 선명도를 유지한다.

- **최종 학습 목표**
  - sparse GT 기반 supervised loss
  - monocular dense supervision 기반 SSI loss
  - gradient regularization
  - 이 세 요소를 결합해 sparse supervision의 한계와 monocular depth의 scale ambiguity를 동시에 보완한다.

### 출력
- RGB와 sparse depth를 입력으로 하는 **dense depth map**
- 더 선명한 경계와 더 일관된 3D 구조를 가진 depth completion 결과

---

## 4. 메모
- 이 논문의 핵심은 backbone 자체보다 **학습 전략(distillation + pre-training + fine-tuning loss 설계)** 에 있다.
- 1단계는 “라벨 없는 자연 이미지도 depth completion 학습에 활용할 수 있다”는 점이 중요하다.
- monocular depth는 dense하지만 metric scale이 부정확할 수 있으므로, 그대로 supervision으로 쓰기보다 **scale/shift invariant하게 정렬해서 사용**하는 것이 핵심이다.
- sparse GT는 절대 스케일을 잡아주고, monocular depth는 조밀한 구조 정보를 제공한다.
- 즉 이 방법은 **sparse LiDAR의 절대 깊이 신뢰성 + monocular foundation model의 dense geometric prior**를 결합한 구조로 볼 수 있다.
- base model로 BP-Net을 사용했지만, 핵심 기여는 BP-Net 자체가 아니라 그 위에 얹힌 **2-stage distillation framework** 다.
- zero-shot 결과와 dense SLAM 적용 예시는, 단순 benchmark 성능 향상뿐 아니라 실제 3D 구조 복원 품질 향상 가능성도 보여준다.

---

## 5. 적용 포인트
 - SSI Loss를 이용한 depth foundation 모델의 결과를 auxiliary loss로 추가하는 concept 
