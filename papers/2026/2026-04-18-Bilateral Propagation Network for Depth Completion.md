# Bilateral Propagation Network for Depth Completion (BP-Net)

- **학회:** CVPR 2024
- **링크:** https://arxiv.org/abs/2403.11270
- **코드:** https://github.com/kakaxi314/BP-Net
- **분야:** Depth Completion, Dense Depth Estimation, 3D Vision

---

## 1. 요약
- 이 논문은 sparse depth map에 바로 convolution을 적용하지 않고, **가장 초기 단계에서 sparse depth를 dense하게 전파**한 뒤 RGB와 융합하고 refinement하는 **3-stage depth completion 네트워크 BP-Net**을 제안한다.
- 핵심은 **bilateral propagation module**로, 주변의 valid depth를 이용해 target pixel의 초기 depth를 만들되, 그 조합 계수 \(\alpha, \beta, \omega\)를 **영상 내용과 공간 거리**에 조건화된 MLP가 동적으로 생성한다는 점이다.
- 전체 구조는 **(1) bilateral propagation → (2) multi-modal fusion(U-Net) → (3) propagation-based refinement(CSPN++ style)** 로 이루어지며, 이를 **coarse-to-fine multi-scale** 방식으로 수행한다.
- NYUv2와 KITTI depth completion에서 강한 성능을 보였고, 특히 초기 단계 전파가 refinement 단계 못지않게 매우 중요하다는 점을 실험으로 보여준다.

---

## 2. 핵심 기여
- sparse depth에 직접 convolution을 적용하는 대신, **전처리 단계에서 depth를 먼저 propagation**하여 희소성 문제를 완화했다.
- **이미지 내용 + 공간 거리**를 함께 활용해 propagation 계수를 생성하는 **학습형 bilateral propagation 모델**을 제안했다.
- bilateral propagation, multi-modal fusion, depth refinement를 **multi-scale end-to-end 구조**로 통합했다.
- ablation study를 통해 **후처리 refinement보다 초기 propagation이 성능 향상에 더 중요할 수 있음**을 보였다.

---

## 3. 방법

### 입력
- 동기화된 **RGB 이미지 \(I\)**
- **Sparse depth map \(S\)**  
  - 보통 LiDAR 또는 SfM 포인트를 이미지 평면에 투영하여 생성

### 핵심 아이디어
- 각 픽셀의 초기 depth \(D'_i\)를 주변의 **가장 가까운 valid sparse depth**로부터 전파한다.
- 이때 단순 보간이 아니라 아래와 같은 **학습형 비선형 조합**을 사용한다.

\[
D'_i = \sum_{j \in \mathcal{N}(i)} \omega_{ij}(\alpha_{ij} S_j + \beta_{ij})
\]

- \(\alpha_{ij}, \beta_{ij}, \omega_{ij}\)는 고정된 값이 아니라, 다음 정보를 입력으로 받는 MLP가 동적으로 생성한다.
  - target pixel의 image encoding
  - source pixel의 image encoding
  - source pixel의 depth encoding
  - source-target 사이의 spatial offset
- 즉, **content-dependent + spatial-aware propagation**을 수행한다.
- 이렇게 생성한 초기 dense depth를 U-Net 기반 **multi-modal fusion**에 넣고, RGB feature와 depth feature를 결합해 **residual depth**를 예측하여 보정한다.
- 이후 **CSPN++ 스타일 refinement**를 통해 affinity map 기반 iterative propagation을 수행하고, sparse depth를 반복적으로 다시 주입해 경계와 구조를 정교하게 다듬는다.
- 이 전 과정을 **6-scale coarse-to-fine 구조**로 수행하며, multi-scale loss로 end-to-end 학습한다.

### 출력
- 최종 **dense depth map \(D\)**

---

## 4. 메모
- 기존 방법은 sparse depth를 CNN에 바로 넣거나, 대략적인 dense depth를 만든 뒤 refinement하는 구조가 많다.
- BP-Net은 **맨 처음부터 sparse depth를 잘 퍼뜨린 뒤** RGB와 융합하기 때문에, 이후 CNN과 refinement가 더 잘 작동한다.
- 논문의 핵심 메시지는  
  **"refinement도 중요하지만, 그보다 먼저 sparse depth를 제대로 densify하는 것이 더 중요하다"**  
  로 볼 수 있다.

---

## 5. 적용 포인트
 - source와 target의 pixel, 거리를 입력으로 받아 filtering 하는 affine 계수를 학습. 즉, bilateral filter의 컨셉(pixel차이와 distance weight)으로 propagation 필터를 hypernetwork mlp로 학습하는 방법
