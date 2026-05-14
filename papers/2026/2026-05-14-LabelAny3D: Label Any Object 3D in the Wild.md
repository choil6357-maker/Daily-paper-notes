# LabelAny3D: Label Any Object 3D in the Wild

- **학회:** NeurIPS 2025
- **링크:** https://arxiv.org/abs/2601.01676
- **코드:** https://github.com/UVA-Computer-Vision-Lab/LabelAny3D
  - 3D BBox refinement interface: https://github.com/UVA-Computer-Vision-Lab/3d_annotator
- **분야:** Open-vocabulary Monocular 3D Object Detection / 3D Auto-labeling / In-the-wild 3D Dataset / Foundation Model 기반 3D Annotation

---

## 1. 요약

- **LabelAny3D**는 단일 RGB 이미지로부터 임의 객체의 **3D bounding box annotation**을 자동 생성하는 파이프라인이다.
- 기존 단안 3D 검출 모델은 Omni3D처럼 실내 또는 자율주행 중심의 데이터셋에 의존해, MS-COCO 같은 **in-the-wild 이미지**에서는 일반화가 어렵다.
- 논문은 이 문제의 핵심 병목을 **대규모 고품질 3D annotation 부족**으로 본다.
- LabelAny3D는 단순히 monocular metric depth만 사용하는 대신, **analysis-by-synthesis** 방식으로 객체별 3D mesh를 복원하고 이를 실제 이미지 장면에 정렬한다.
- 전체 흐름은 **초해상화 → segmentation 정제 → amodal completion → object 3D reconstruction → scene depth estimation → 2D-3D alignment → scale alignment → 3D box fitting**이다.
- 객체 shape은 TRELLIS 같은 image-to-3D 모델로 복원하고, 장면 구조는 MoGe의 relative depth와 Depth Pro의 metric depth를 결합해 추정한다.
- 복원된 객체 mesh는 MASt3R 기반 2D matching과 PnP-RANSAC을 통해 실제 이미지의 camera coordinate frame으로 정렬된다.
- 최종적으로 mesh surface point cloud에 PCA를 적용하여 yaw를 추정하고, tight 3D bounding box를 생성한다.
- 논문은 이 파이프라인으로 MS-COCO 기반의 새로운 open-vocabulary monocular 3D detection benchmark인 **COCO3D**를 구축한다.
- 실험 결과, LabelAny3D pseudo label은 기존 OVM3D-Det 대비 annotation 품질이 높고, OVMono3D fine-tuning 시 COCO3D 및 novel category 성능 향상에 기여한다.

---

## 2. 핵심 기여

- **LabelAny3D 자동 3D 라벨링 파이프라인 제안**
  - 단일 RGB 이미지에서 임의 카테고리 객체의 3D bounding box를 생성한다.
  - 기존 방식처럼 metric depth와 object size prior에만 의존하지 않고, object-centric 3D reconstruction과 scene geometry alignment를 결합한다.
  - in-the-wild 이미지에서도 더 다양한 객체 크기와 형태를 다룰 수 있다.

- **COCO3D 벤치마크 구축**
  - MS-COCO validation set을 기반으로 만든 open-vocabulary monocular 3D detection benchmark이다.
  - 2,039장의 human-refined image와 5,373개의 instance를 포함한다.
  - MS-COCO 80개 category 전반을 대상으로 하되, 평가 instance가 너무 적거나 aspect ratio가 극단적인 일부 category는 evaluation에서 제외한다.

- **단안 3D 검출 학습 성능 개선**
  - LabelAny3D로 생성한 pseudo label을 사용해 OVMono3D의 lifting head를 학습 또는 fine-tuning한다.
  - Omni3D만으로 pretrained된 모델보다 COCO3D에서 더 좋은 일반화 성능을 보인다.
  - 기존 OVM3D-Det pseudo label 대비 label noise가 적고, downstream 3D detection 성능 향상 효과가 크다.

---

## 3. 방법

### 입력

- 단일 RGB 이미지
- 2D instance segmentation mask
  - MS-COCO annotation을 기반으로 하되, 더 정제된 COCONut mask 사용
- 객체 category label
- 사용 모델 및 모듈
  - **InvSR:** image super-resolution
  - **COCONut:** refined 2D segmentation mask
  - **Gen3DSR:** amodal completion
  - **TRELLIS:** single-view object 3D reconstruction
  - **MoGe:** affine-invariant / relative scene geometry estimation
  - **Depth Pro:** metric depth estimation
  - **MASt3R:** dense correspondence matching
  - **PnP + RANSAC:** object pose estimation
  - **PCA:** 3D box yaw estimation

### 핵심 아이디어

- **문제 정의**
  - 단일 RGB 이미지에서 3D bounding box를 직접 예측하거나, metric depth만으로 box를 추정하면 in-the-wild 환경에서 불안정하다.
  - 객체 크기, camera focal length, 실제 거리, occlusion이 서로 얽혀 있어 monocular metric depth는 근본적으로 ill-posed하다.

- **핵심 접근**
  - LabelAny3D는 먼저 객체별 3D mesh를 복원한 뒤, 그 mesh를 실제 장면의 2D 이미지 및 depth 구조와 맞춘다.
  - 즉, 단순 depth lifting이 아니라 **“3D 객체를 만들고, 렌더링하고, 실제 이미지와 맞도록 정렬하는” analysis-by-synthesis 방식**이다.

- **세부 파이프라인**

  1. **Image Super-resolution**
     - MS-COCO 이미지의 작은 객체와 compression artifact 문제를 완화하기 위해 InvSR로 이미지를 4배 초해상화한다.
     - 목적은 객체 boundary와 fine detail을 개선하여 reconstruction 및 matching 성능을 높이는 것이다.

  2. **2D Instance Segmentation 정제**
     - COCONut의 refined mask를 사용한다.
     - 이미지 boundary와 많이 겹치는 truncated object는 제거한다.
     - mask를 super-resolution 이미지 크기에 맞게 nearest-neighbor interpolation으로 upscaling한다.
     - mask area가 너무 작은 객체는 reliable geometry 추정이 어렵기 때문에 제외한다.

  3. **Amodal Completion**
     - occlusion된 객체 crop을 Gen3DSR의 amodal completion diffusion model로 보완한다.
     - 보이는 영역뿐 아니라 가려진 부분까지 복원된 객체 crop을 만든다.

  4. **Single-view 3D Object Reconstruction**
     - 완성된 객체 crop을 TRELLIS에 입력하여 3D mesh를 생성한다.
     - 출력 mesh는 canonical pose와 normalized scale을 가진다.
     - 이 단계에서는 아직 실제 장면 내 위치와 metric scale은 알 수 없다.

  5. **Scene Geometry Estimation**
     - MoGe로 relative scene geometry를 추정한다.
     - Depth Pro로 metric depth를 추정한다.
     - MoGe depth를 Depth Pro의 metric scale과 perspective에 맞춰 alignment한다.
     - 정렬된 depth와 camera intrinsic을 사용해 scene point cloud를 복원한다.

  6. **Pose Estimation via 2D-3D Alignment**
     - 복원된 object mesh를 여러 시점에서 렌더링한다.
     - MASt3R로 real image와 rendered view 사이의 dense 2D-2D correspondence를 구한다.
     - rendered depth와 rendering camera parameter를 이용해 rendered pixel을 mesh 위 3D point로 unproject한다.
     - 최종적으로 3D mesh point와 real image pixel 사이의 3D-2D correspondence를 얻는다.
     - PnP + RANSAC으로 object pose `(R, T)`를 추정한다.

  7. **Scale Estimation via Depth Alignment**
     - segmentation mask와 rendered mask의 overlap 영역을 계산한다.
     - real depth와 rendered depth의 median ratio로 scale factor `s`를 추정한다.
     - 추정된 scale을 rotation 및 translation과 결합해 object mesh를 metric-scale scene에 배치한다.

  8. **3D Annotation Generation**
     - 배치된 object mesh 표면에서 point cloud를 균일 샘플링한다.
     - TRELLIS의 canonical upright direction을 gravity direction으로 간주한다.
     - point cloud를 horizontal plane에 투영하고 PCA로 dominant yaw를 추정한다.
     - point cloud 전체를 감싸는 tight 3D bounding box를 fitting한다.

- **학습 적용**
  - LabelAny3D pseudo label을 사용해 OVMono3D를 학습한다.
  - OVMono3D는 open-vocabulary 2D detector로 2D box를 얻은 뒤, class-agnostic lifting head로 3D cuboid를 예측한다.
  - 학습 loss는 3D attribute별 disentangled loss와 전체 3D box Chamfer loss를 결합한다.

### 출력

- 객체별 3D bounding box annotation
  - center position
  - dimensions: width, height, length
  - orientation / yaw
  - depth
  - category label
- metric-scale scene에 정렬된 object mesh
- COCO3D benchmark
  - human refinement가 적용된 evaluation set
  - LabelAny3D pseudo label 기반 training set
- downstream detector 학습용 pseudo 3D annotations

---

## 4. 메모

- 이 논문의 핵심은 **monocular 3D detection 모델 자체보다, 학습과 평가에 사용할 수 있는 3D annotation을 어떻게 대규모로 만들 것인가**에 있다.
- 기존 OVM3D-Det은 metric depth와 object size prior에 크게 의존한다. 이 방식은 car, pedestrian처럼 크기가 비교적 일정한 객체에는 유효하지만, elephant, child, boat처럼 intra-class size variation이 큰 객체에서는 취약하다.
- LabelAny3D는 object mesh reconstruction과 relative depth alignment를 사용하기 때문에, appearance와 shape에 더 일관적인 3D box를 만들 수 있다.
- Ablation 결과상 중요한 구성 요소는 다음과 같다.
  - InvSR super-resolution: 작은 객체와 먼 객체 detail 복원에 중요
  - amodal completion: occlusion 완화에 중요
  - MoGe relative depth: Depth Pro 단독 사용보다 안정적인 relative layout 제공
  - TRELLIS: DreamGaussian보다 더 높은 품질의 object mesh 생성
  - MASt3R + PnP: ICP보다 robust한 object-scene alignment 제공
- 한계도 명확하다.
  - heavy occlusion, textureless region, small object에서는 foundation model들이 실패할 수 있다.
  - TRELLIS가 viewing direction 방향으로 depth가 애매한 mesh를 만들면 RGBD point cloud와 misalignment가 발생할 수 있다.
  - COCO3D는 benchmark 신뢰성을 위해 truncation, severe occlusion, erroneous depth sample을 제외하므로 exhaustive annotation dataset은 아니다.
- 사용자의 목적이 **집안 물체의 3D cuboid/volume annotation 생성**이라면, LabelAny3D 구조는 참고 가치가 크다.
  - 다만 실제 robot vacuum view에서는 MS-COCO보다 viewpoint가 낮고 occlusion이 많으며, 바닥/벽/가구 scale이 중요하므로 depth alignment와 camera intrinsic calibration을 더 엄격하게 검증해야 한다.
  - Kinect/RGB-D 또는 SLAM reconstruction이 있다면, LabelAny3D의 monocular depth 대신 실제 depth/TSDF mesh를 결합하는 방향이 더 안정적일 수 있다.

---

## 5. 적용 포인트
 - 3D cuboid detection을 위한 pseudo labeling 방식으로 활용가능
 - 제공하는 3D annotation tool도 활용 가능
