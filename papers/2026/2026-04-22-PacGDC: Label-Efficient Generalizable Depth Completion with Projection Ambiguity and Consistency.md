# PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency

- **학회:** ICCV 2025
- **링크:** https://arxiv.org/abs/2507.07374
- **코드:** https://github.com/Wang-xjtu/PacGDC
- **분야:** Depth Completion, Generalizable Depth Completion, Monocular Depth Estimation, Label-Efficient Learning

---

## 1. 요약
- PacGDC는 **일반화 가능한 depth completion**을 위해, 적은 GT 라벨만으로도 학습 데이터 다양성을 크게 늘리는 **label-efficient 데이터 합성 방법**이다.
- 핵심 출발점은 **2D-to-3D projection ambiguity**이다. 같은 2D 이미지라도 여러 가능한 3D geometry가 대응될 수 있다는 점을 이용한다.
- 논문은 이 모호성을 **shape ambiguity**와 **position ambiguity**로 분해하고, 동시에 이미지와 sparse depth가 각각 shape/position을 제약하는 **projection consistency**를 강조한다.
- 이를 바탕으로, 하나의 시각 장면에 대해 여러 개의 **pseudo dense depth label**을 합성하여 학습 데이터 coverage를 확장한다.
- pseudo depth는 **DepthAnything, DepthPro** 같은 monocular depth foundation model의 출력을 활용해 생성한다.
- 추가로 **interpolation**과 **relocation** 전략을 적용해 local/global scene scale과 위치 다양성을 더 키운다.
- labeled 데이터뿐 아니라 **unlabeled image**도 합성 파이프라인에 포함해 데이터 다양성을 더욱 확장한다.
- 최종적으로 합성된 dense depth에서 sparse depth를 다시 샘플링하여 **pseudo triplet (image, sparse depth, dense depth)** 을 만든다.
- 이 pseudo triplet으로 **SPNet** 기반 depth completion 모델을 학습하며, 추론 시에는 추가 연산 비용이 없다.
- 실험 결과, PacGDC는 **zero-shot / few-shot depth completion** 모두에서 강한 일반화 성능을 보이며 SOTA 수준 결과를 달성한다.

---

## 2. 핵심 기여
- **2D-to-3D projection ambiguity와 consistency를 depth completion 데이터 합성에 활용**하여, 추가 real label 없이도 geometry diversity를 크게 늘렸다.
- **여러 depth foundation model + interpolation + relocation + unlabeled image**를 결합한 새로운 pseudo depth synthesis pipeline을 제안했다.
- 이 데이터 중심 접근을 통해 **zero-shot generalization**과 **few-shot adaptation** 모두에서 강한 성능을 입증했다.

---

## 3. 방법

### 입력
- **입력 이미지** \(I\)
- **sparse depth map** \(p\)
- 학습 시 일부 데이터에 대해 **ground-truth dense depth** \(d\)
- 추가적으로 **unlabeled image** \(I^u\)
- monocular depth foundation model 출력 (예: DepthAnything, DepthPro)

### 핵심 아이디어
- 기본 depth completion은 \((I, p) \rightarrow d\) 문제로 정의된다.
- PacGDC는 한 장의 이미지에 대해 여러 개의 pseudo dense depth \(\hat d\)를 합성해 **원래 triplet을 대체/확장하는 pseudo triplet**을 만든다.
- 이때 이론적 기반은 다음 두 가지다.
  - **Projection ambiguity:** 같은 2D 이미지가 여러 가능한 3D geometry에 대응될 수 있음
  - **Projection consistency:** 이미지 semantic은 shape를, sparse depth는 position/scale을 제약함
- depth foundation model은 shape/semantic은 비교적 잘 맞추지만 scale은 부정확할 수 있으므로, 이를 오히려 **scene scale diversity 생성 수단**으로 활용한다.
- pseudo depth는 다음 방식으로 생성된다.
  - foundation model prediction 사용
  - GT depth와 **interpolation**
  - spatial arrangement를 바꾸는 **relocation**
  - 여러 foundation model 출력을 혼합
- 이후 합성된 dense depth \(\hat d\)에서 sparse depth \(\hat p\)를 다시 샘플링해, 입력-정답 간 consistency를 유지한다.
- unlabeled image에도 같은 파이프라인을 적용해, semantic 및 scale 다양성을 추가 확보한다.
- 최종 학습은 SPNet 위에서 수행되며, 핵심은 **모델 구조 변화보다 데이터 다양성 확대**에 있다.

### 출력
- 합성된 **pseudo dense depth label** \(\hat d\)
- \(\hat d\)에서 샘플링된 **pseudo sparse depth** \(\hat p\)
- 최종 학습용 **pseudo triplet** \((I, \hat p, \hat d)\)
- 이를 이용해 학습된 **generalizable depth completion model**

---

## 4. 메모
- 이 논문의 핵심은 **foundation model의 scale 오류를 약점이 아니라 데이터 다양성 자원으로 활용했다**는 점이다.
- 단순히 같은 이미지에 여러 정답을 주는 것이 아니라, 각 pseudo depth에서 다시 sparse depth를 샘플링해 **조건부 문제**로 만든 것이 포인트다.
- 따라서 모델은 이미지 semantic과 sparse depth anchor를 함께 이용해 depth를 복원하도록 학습된다.
- 다만 pseudo depth가 너무 부정확하거나 sparse depth가 지나치게 희소하면, 평균화/과도한 smoothing 위험이 있을 수 있다.
- labeled branch에서는 GT depth가 scale anchor 역할을 하지만, unlabeled branch에서는 absolute metric scale 보장이 약하다.
- 그럼에도 논문은 **self-consistency와 다양성 증가**가 generalization 향상에 더 중요하다고 본다.
- 실험적으로 zero-shot에서는 unseen dataset들에 대해 강한 성능을 보였고, few-shot에서는 매우 적은 KITTI 샘플만으로도 full-shot baseline에 근접하거나 일부를 능가했다.
- ablation 결과, **foundation model 추가 → interpolation → relocation → multi-model → unlabeled image** 순으로 성능이 꾸준히 좋아졌다.

---

## 5. 적용 포인트
 - Depth completion의 GT 생성방식으로 활용 가능
