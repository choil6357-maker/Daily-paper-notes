# ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

- **학회:** CVPR 2023
- **링크:** https://arxiv.org/abs/2301.00808
- **코드:** https://github.com/facebookresearch/ConvNeXt-V2
- **분야:** Self-Supervised Learning, Masked Image Modeling, ConvNet Architecture, Visual Representation Learning

---

## 1. 요약
- 본 논문은 ConvNet 계열 모델인 ConvNeXt를 masked autoencoder 기반 자기지도학습에 더 적합하게 만들기 위해 **FCMAE(Fully Convolutional Masked AutoEncoder)**와 **GRN(Global Response Normalization)**을 함께 제안한다.
- 기존 MAE는 ViT처럼 patch token sequence를 다루는 Transformer 구조에 잘 맞지만, ConvNet은 dense sliding window 방식으로 2D grid 전체를 처리하기 때문에 masked patch를 encoder에서 완전히 제거하기 어렵다.
- 이를 해결하기 위해 저자들은 masked image를 **2D sparse data**로 보고, encoder의 convolution을 **submanifold sparse convolution**으로 바꾸어 visible patch만 처리하도록 설계한다.
- FCMAE는 random masking, sparse ConvNeXt encoder, lightweight ConvNeXt decoder, masked patch reconstruction loss로 구성된다.
- 하지만 FCMAE만 적용한 ConvNeXt V1은 supervised ConvNeXt 최고 성능을 확실히 넘지 못했으며, feature collapse 문제가 관찰되었다.
- 이를 해결하기 위해 GRN layer를 도입하여 channel 간 global response competition을 강화하고 feature diversity를 높인다.
- GRN은 각 channel feature map의 L2-norm response를 계산하고, 전체 channel response 대비 상대적 중요도를 이용해 feature를 보정한다.
- ConvNeXt V2는 ConvNeXt block에 GRN을 추가하고 LayerScale을 제거한 새로운 model family이다.
- FCMAE와 GRN을 함께 사용할 때 성능 향상이 가장 크게 나타났으며, 이는 self-supervised learning에서는 architecture와 pre-training framework를 함께 설계해야 함을 보여준다.
- ConvNeXt V2는 ImageNet classification, COCO object detection, ADE20K semantic segmentation에서 기존 ConvNeXt V1 및 Swin Transformer와 경쟁하거나 더 좋은 성능을 보인다.

---

## 2. 핵심 기여
- **ConvNet을 위한 Fully Convolutional MAE 제안**
  - ViT 기반 MAE를 ConvNet에 그대로 적용하는 대신, masked image를 sparse 2D grid로 해석하고 sparse convolution을 사용하여 visible patch만 encoder에서 처리한다.
  - 이를 통해 masked region의 information leakage와 train/test mismatch 문제를 줄인다.

- **Global Response Normalization(GRN) 제안**
  - FCMAE로 ConvNeXt를 사전학습할 때 발생하는 feature collapse 문제를 분석하고, channel 간 feature competition을 강화하는 GRN layer를 제안한다.
  - GRN은 L2-norm 기반 global response aggregation, divisive normalization, feature calibration으로 구성된다.

- **ConvNeXt V2 model family 구축**
  - 기존 ConvNeXt block에 GRN을 추가하고 LayerScale을 제거하여 self-supervised masked pre-training에 적합한 ConvNet 구조를 만든다.
  - Atto부터 Huge까지 다양한 모델 크기로 확장 가능하며, ImageNet/COCO/ADE20K에서 강한 scaling behavior를 보인다.

---

## 3. 방법

### 입력
- 입력은 일반 RGB 이미지이다.
- 사전학습에서는 ImageNet-1K 이미지를 사용하며, 모델 확장 실험에서는 ImageNet-22K intermediate fine-tuning도 수행한다.
- 입력 이미지는 random resized crop 정도의 최소 augmentation만 적용된다.
- masking은 32×32 patch 단위로 수행하며, masking ratio는 0.6이다.
- downstream task에서는 ImageNet classification, COCO object detection/instance segmentation, ADE20K semantic segmentation을 평가한다.

### 핵심 아이디어
- **문제 설정**
  - MAE는 Transformer 구조에서는 visible token만 encoder에 넣을 수 있어 masked region 정보 누설을 막기 쉽다.
  - 반면 ConvNet은 2D grid 전체에 convolution을 적용하므로, masked token이나 zero-filled region이 encoder 내부에서 계속 연산에 참여할 수 있다.
  - 이로 인해 masked image modeling에서 shortcut learning, information leakage, train/test mismatch가 발생할 수 있다.

- **FCMAE**
  - masked image를 dense image가 아니라 **visible 위치만 존재하는 2D sparse array**로 본다.
  - 사전학습 중 ConvNeXt encoder의 standard convolution을 **submanifold sparse convolution**으로 변환한다.
  - submanifold sparse convolution은 active coordinate를 확장하지 않기 때문에, masked 위치가 encoder 중간에서 다시 활성화되지 않는다.
  - encoder는 visible patch 위치만 처리하고, masked patch는 decoder가 복원하도록 한다.
  - decoder는 single ConvNeXt block으로 구성된 lightweight fully convolutional decoder이며, decoder dimension은 512이다.
  - reconstruction target은 patch-wise normalized original image이고, loss는 masked patch에 대해서만 MSE로 계산한다.

- **Sparse Convolution의 역할**
  - 일반 dense convolution은 masked 위치도 grid에 남겨두고 연산하므로 masked region 정보가 주변 feature를 통해 섞일 수 있다.
  - sparse convolution은 visible 위치만 active coordinate로 두고 연산한다.
  - 특히 submanifold sparse convolution은 output active 위치를 input active 위치와 동일하게 유지하므로, masked 위치가 encoder에서 계속 비활성 상태로 유지된다.
  - pre-training 후 fine-tuning 단계에서는 sparse convolution weight를 standard dense convolution으로 되돌려 downstream task에 사용한다.

- **Feature Collapse 분석**
  - FCMAE를 ConvNeXt V1에 적용하면 dimension-expansion MLP layer에서 dead/saturated feature map이 많아지고 channel 간 feature redundancy가 증가한다.
  - 저자들은 channel feature map 간 평균 pair-wise cosine distance를 계산해 feature diversity를 정량적으로 분석한다.
  - FCMAE pre-trained ConvNeXt V1은 feature cosine distance가 낮아지는 경향을 보이며, 이는 feature collapse를 의미한다.

- **GRN**
  - GRN은 channel 간 response competition을 강화하기 위한 normalization layer이다.
  - 입력 feature \(X \in \mathbb{R}^{H \times W \times C}\)에 대해 각 channel \(X_i\)의 L2-norm response를 계산한다.
  - 각 channel response를 전체 channel response 합으로 나누어 상대적 중요도를 구한다.
  - 이 normalized response를 원래 feature에 곱해 feature를 calibration한다.
  - 실제 구현에서는 learnable parameter \(\gamma\), \(\beta\)와 residual connection을 추가해 학습 초기에 identity function처럼 동작하도록 한다.
  - 최종 형태는 다음과 같다.

  ```text
  X_i = gamma * X_i * N(G(X)_i) + beta + X_i
  ```

- **ConvNeXt V2 Block**
  - ConvNeXt V2는 기존 ConvNeXt block에 GRN을 추가한 구조이다.
  - GRN 적용 시 LayerScale은 필요하지 않아 제거한다.
  - 구조적으로는 기존 ConvNeXt의 장점을 유지하면서, masked self-supervised pre-training에 더 적합한 channel response normalization을 추가한 형태이다.

### 출력
- 사전학습 단계의 출력은 masked patch를 복원한 reconstructed image이다.
- 학습 loss는 masked patch에 대한 patch-normalized MSE이다.
- 사전학습 후 얻는 최종 출력은 다양한 downstream task에 사용할 수 있는 ConvNeXt V2 backbone weight이다.
- fine-tuning 단계에서는 task에 따라 다음 출력을 생성한다.
  - ImageNet: image classification logits
  - COCO: object detection box prediction 및 instance segmentation mask
  - ADE20K: semantic segmentation map

---

## 4. 메모
- ConvNeXt V2의 핵심은 단순히 ConvNeXt에 MAE를 붙인 것이 아니라, **MAE에 맞게 ConvNet 구조도 함께 수정했다는 점**이다.
- FCMAE는 ConvNet에서도 ViT-MAE처럼 encoder가 visible patch만 보도록 만드는 것이 핵심이다.
- sparse convolution은 단순 효율화 기법이 아니라 masked region information leakage를 막는 핵심 알고리즘 요소이다.
- 논문 실험에서 sparse convolution이 없으면 top-1 accuracy가 79.3까지 떨어지고, sparse convolution을 사용하면 83.7까지 오른다.
- FCMAE만으로는 supervised ConvNeXt V1 최고 성능을 명확히 넘지 못하며, 이 한계를 해결하는 요소가 GRN이다.
- GRN은 추가 parameter/FLOPs overhead 없이 feature diversity를 높이는 단순한 layer이다.
- GRN은 supervised setting에서는 효과가 작지만, FCMAE와 결합했을 때 큰 성능 향상을 만든다.
- 이는 self-supervised learning에서 architecture와 training objective를 따로 설계하면 최적 성능을 얻기 어렵다는 점을 보여준다.
- ConvNeXt V2 + FCMAE는 ImageNet, COCO, ADE20K에서 모두 강한 transfer 성능을 보인다.
- 이 논문은 ConvNet도 Transformer처럼 대규모 masked image modeling 기반 representation learning에서 경쟁력 있을 수 있음을 보여주는 사례로 볼 수 있다.

---

## 5. 적용 포인트
 - swin 대비 속도 가속 측면에서 이득이 있으면서 성능도 더 좋은 편. 
