# 제목: Scale Propagation Network for Generalizable Depth Completion


- **학회:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
- **링크:** https://arxiv.org/abs/2410.18408
- **코드:** https://github.com/Wang-xjtu/SPNet
- **분야:** Depth Completion, Generalizable Depth Completion, 3D Vision

---

## 1. 요약
- 이 논문은 **generalizable depth completion**, 즉 학습 때 보지 못한 새로운 장면에서도 잘 동작하는 depth completion 문제를 다룬다.
- 저자들은 기존 depth completion 네트워크의 핵심 병목이 **정규화 층(BN/IN/LN)** 에 있다고 분석한다.
- 일반 이미지 인식에서는 정규화가 scale invariance를 만드는 데 유리하지만, depth completion에서는 오히려 **장면의 scale 정보 자체를 보존하고 전달하는 것**이 중요하다.
- 특히 sparse depth map의 scale이 출력 dense depth의 scale을 결정해야 하는데, 기존 normalization은 이 scale propagation을 방해한다고 본다.
- 이를 해결하기 위해 저자들은 **SP-Norm (Scale Propagation Normalization)** 을 제안한다.
- SP-Norm은 normalized feature에서 SLP로 multiplier를 만들고, 이를 원래 입력 feature에 **곱하는 방식**으로 입력 scale 정보를 출력까지 전달한다.
- 이 방식은 normalization operator를 유지하므로 학습 안정성은 살리면서도, 기존 normalization보다 scale propagation 특성을 더 잘 만족하도록 설계되었다.
- 전체 네트워크는 **SP-Norm + ConvNeXt V2 기반 encoder-decoder 구조**로 구성되며, LN을 SP-Norm으로 치환하고 GRN을 제거하며 GELU를 ReLU로 바꾼다.
- 학습은 Matterport3D, HRWSI, vKITTI, UnrealCV의 혼합 데이터셋으로 수행하고, 평가는 Ibims, KITTI, NYUv2, DIODE, ETH3D, Sintel의 **6개 unseen dataset**에서 수행한다.
- 결과적으로 제안 모델은 다양한 sparse depth 조건(랜덤 샘플링, LiDAR line, structured-light hole)에서 기존 SOTA 대비 **더 나은 정확도와 더 빠른 추론, 더 낮은 메모리 사용량**을 보인다.

---

## 2. 핵심 기여
- 기존 normalization layer가 generalizable depth completion에서 **입력 sparse depth의 scale을 출력으로 잘 전달하지 못한다**는 점을 분석했다.
- 입력에서 출력으로의 scale propagation을 가능하게 하면서도 normalization operator를 유지하는 **SP-Norm**을 제안했다.
- **SP-Norm + ConvNeXt V2** 기반의 새로운 depth completion 네트워크를 설계하고, basic block과 전체 architecture를 효율적으로 재구성했다.
- 6개의 unseen dataset과 다양한 sparse depth 조건에서 실험하여, 제안 방식이 **정확도·속도·메모리 측면에서 강한 일반화 성능**을 보인다는 점을 입증했다.

---

## 3. 방법

### 입력
- RGB 이미지 1장
- Sparse depth map 1장
  - 랜덤 샘플링된 sparse depth (0.1% / 1% / 10% valid pixels)
  - 4 / 8 / 16 / 32 / 64-line LiDAR sparse points
  - Structured-Light sensor의 hole이 있는 raw depth map

### 핵심 아이디어
- **문제 정의:** unseen scene에서 depth completion이 잘 안 되는 이유를, 단순히 데이터셋 차이보다도 **네트워크 내부의 normalization 설계 문제**로 본다.
- **SP-property 정의:** generalizable depth completion에서는 출력 dense depth `z`가 입력 sparse depth `d`와 scale 측면에서 비례 관계를 유지해야 한다고 본다. 즉, 입력 depth가 `s`배 되면 출력 depth도 비슷하게 `s`배 되어야 한다.
- **기존 normalization의 한계:** BN/IN/LN은 입력을 평균/분산으로 정규화한 뒤 affine factor로 다시 변환하는 구조라서, 테스트 시 scene마다 달라지는 scale 정보를 안정적으로 복원하기 어렵다.
- **SP-Norm:** 정규화된 feature를 SLP에 넣어 multiplier를 만들고, 이 multiplier를 **원래 입력 feature에 곱해주는 구조**를 사용한다. 즉, 입력 scale 정보를 완전히 지우지 않고 보존하면서 modulation한다.
- **수식 관점:** 기존 normalization은 출력 통계량이 주로 affine parameter에 의해 결정되는 반면, SP-Norm은 출력의 평균/분산이 입력의 평균/분산에 비례하도록 학습 가능하게 설계된다.
- **Basic block 변경점:** ConvNeXt V2 block에서 LN→SP-Norm으로 교체, GRN 제거, GELU→ReLU 변경.
- **Architecture:** heavyweight encoder + lightweight decoder 구조를 사용해 receptive field와 효율을 동시에 확보한다.
- **Data augmentation:** scale generalization을 위해 Resized-Crop과 Depth-Scaling을 사용해 장면/깊이 scale 다양성을 인위적으로 늘린다.
- **Loss:** scale-adaptive loss와 multi-scale scale-invariant gradient loss를 함께 사용한다.

### 출력
- 입력 sparse depth map을 dense depth map으로 복원한 **완성 depth map**
- 목표는 seen dataset 성능보다도, **unseen scene에서도 scale이 잘 맞는 robust depth completion**을 얻는 것

---

## 4. 메모
- 이 논문의 가장 중요한 메시지는 **"depth completion에서는 scale invariance가 아니라 scale propagation이 중요하다"** 는 점이다.
- 즉, 일반 vision backbone에서 좋은 normalization 설계가 depth completion 일반화에는 오히려 독이 될 수 있다는 주장이다.
- SP-Norm은 normalization을 완전히 제거하지 않고 유지한다는 점에서, ReZero/Fixup 같은 non-normalization 접근보다 실용적이다.
- ConvNeXt V2를 그대로 쓰지 않고, depth task에 맞게 **GRN 제거**와 **SP-Norm 치환**을 했다는 점이 핵심이다.
- 실험적으로 tiny 모델도 속도와 메모리 측면에서 강점이 크며, large 모델로 갈수록 성능이 안정적으로 올라가 확장성도 확인했다.
- 저자들은 결론에서, 현재 SP-Norm의 입력-출력 scale 비율은 테스트 시 **상수적 성격**을 가지므로, 향후에는 self-attention의 dynamic weight처럼 **dynamic SP-Norm** 방향으로 확장할 수 있다고 언급한다.
- 따라서 후속 아이디어로는 scene-conditioned multiplier, attention 기반 scale modulation, dynamic prompt/conditioning 기반 depth completion 등으로 연결해 볼 수 있다.
- 구현 관점에서는 "normalization을 feature whitening 용이 아니라 scale-preserving modulation 용으로 다시 설계했다"고 이해하면 된다.

---

## 5. 적용 포인트
 - SP-Norm과 Resized-Crop, Depth-Scaling augmentation 방법, 그리고  self-attention의 dynamic weight처럼 **dynamic SP-Norm** 방향의 연구
