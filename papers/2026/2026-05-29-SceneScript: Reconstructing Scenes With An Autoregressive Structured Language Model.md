# SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model

- **학회:** ECCV 2024
- **링크:** https://arxiv.org/abs/2403.13064
- **코드:** https://github.com/facebookresearch/scenescript
- **분야:** 3D Scene Reconstruction, Indoor Layout Estimation, 3D Object Detection, Structured Language Modeling, Parametric Scene Representation

---

## 1. 요약

- **SceneScript**는 실내 장면을 mesh, voxel, point cloud, NeRF 같은 dense representation으로 직접 복원하지 않고, `make_wall`, `make_door`, `make_window`, `make_bbox` 같은 **구조화된 언어 명령어(sequence of structured language commands)** 로 표현하는 방법이다.
- 입력은 실내 공간을 walkthrough한 **video stream / posed images / SLAM 기반 point cloud**이며, 출력은 장면을 재구성할 수 있는 **tokenized scene program**이다.
- 모델 구조는 **encoder-decoder architecture**이다. Encoder는 point cloud 또는 posed image sequence를 latent scene code로 변환하고, Transformer decoder는 이를 기반으로 SceneScript token을 autoregressive하게 생성한다.
- 기본 command는 벽, 문, 창문을 표현하는 layout command와, 가구/객체를 oriented 3D bounding box로 표현하는 `make_bbox` command로 구성된다.
- 추가 command를 정의하면 object part primitive, curved wall, door opening state 등으로 쉽게 확장할 수 있다.
- 학습을 위해 Meta는 **Aria Synthetic Environments, ASE**라는 10만 개 규모의 synthetic indoor scene dataset을 생성했다.
- SceneScript는 architectural layout estimation에서 기존 baseline보다 높은 성능을 보였고, 3D object detection에서도 전문 detection model과 경쟁력 있는 성능을 보였다.

---

## 2. 핵심 기여

- **구조화 언어 기반 3D scene representation 제안**
  - 장면을 dense geometry가 아니라 명령어 시퀀스로 표현한다.
  - 예: `make_wall(...)`, `make_door(...)`, `make_window(...)`, `make_bbox(...)`.
  - 출력이 compact하고, 해석 가능하며, 편집 가능하다.

- **Autoregressive Transformer 기반 scene-to-program generation**
  - Scene observation을 encoder로 latent scene code로 변환한다.
  - Transformer decoder가 GPT처럼 다음 token을 순차적으로 예측한다.
  - 최종 token sequence를 interpreter가 parsing하여 3D layout으로 변환한다.

- **Layout estimation과 object detection을 하나의 language interface로 통합**
  - 벽/문/창문 같은 architectural layout과 sofa/table/chair 같은 3D object bbox를 동일한 token prediction framework에서 생성한다.
  - 별도의 task-specific head 없이 command를 추가하는 방식으로 기능을 확장할 수 있다.

- **Aria Synthetic Environments 데이터셋 공개**
  - 10만 개의 synthetic indoor scene을 제공한다.
  - 각 scene은 egocentric walkthrough, photorealistic rendering, depth, instance segmentation, SceneScript GT command sequence를 포함한다.

- **Command 확장성 검증**
  - `make_prim` command를 추가하여 table/chair/sofa 같은 객체를 cuboid 또는 extruded cylinder primitive로 coarse reconstruction할 수 있음을 보였다.
  - 향후 curved wall, door open state, Blender geometry nodes 기반 parametric model 등으로 확장 가능성을 제시했다.

---

## 3. 방법

### 입력

- **Posed image sequence**
  - 실내 공간을 이동하면서 촬영한 egocentric video frame.
  - 각 frame은 camera pose를 가진다.
  - SLAM output으로 얻을 수 있는 형태이다.

- **Point cloud**
  - SLAM, SfM, RGB-D, LiDAR 등으로 생성된 3D point cloud.
  - 논문에서는 Project Aria의 visual-inertial SLAM 기반 semi-dense point cloud를 사용한다.

- **Point cloud + lifted image features**
  - 각 3D point를 여러 posed RGB image에 projection한다.
  - 해당 pixel 위치의 CNN image feature를 가져와 3D point feature에 붙인다.
  - geometry와 appearance/semantic cue를 함께 사용하는 방식이다.

---

### 핵심 아이디어

#### 1) Scene을 command sequence로 표현

SceneScript는 하나의 scene을 다음과 같은 sequence로 표현한다.

```text
START
PART MAKE_WALL ...
PART MAKE_WALL ...
PART MAKE_DOOR ...
PART MAKE_WINDOW ...
PART MAKE_BBOX ...
STOP
```

각 `PART`는 하나의 entity command를 구분한다.  
명령어 개수는 고정되어 있지 않으며, scene complexity에 따라 sequence 길이가 달라질 수 있다.

---

#### 2) 기본 command 정의

실내 architectural layout을 위해 세 가지 command를 사용한다.

```text
make_wall(...)
make_door(...)
make_window(...)
```

- `make_wall`: 중력 방향에 정렬된 2D wall plane을 표현한다.
- `make_door`: wall에 존재하는 box-shaped cutout 형태의 문을 표현한다.
- `make_window`: wall에 존재하는 box-shaped cutout 형태의 창문을 표현한다.

객체는 다음 command로 표현한다.

```text
make_bbox:
    id,
    class,
    position_x,
    position_y,
    position_z,
    angle_z,
    scale_x,
    scale_y,
    scale_z
```

이는 중력 방향에 정렬된 oriented 3D bounding box를 의미한다.  
즉, 대부분의 가구처럼 바닥에 놓인 물체를 `center + yaw + size`로 표현한다.

---

#### 3) Encoder

논문은 세 가지 encoder variant를 실험한다.

##### A. Point Cloud Encoder

```text
P ∈ R^{N x 3}
F_geo = E_geo(P)
F_geo ∈ R^{K x 512}, K << N
```

처리 순서:

1. point cloud를 5cm resolution으로 discretize한다.
2. sparse 3D convolution을 적용한다.
3. down convolution으로 point 수를 줄인다.
4. K개의 latent geometry feature를 만든다.
5. 각 feature에 active site coordinate를 붙여 positional information을 추가한다.
6. 이 feature sequence를 decoder에 전달한다.

##### B. Point Cloud + Lifted Image Feature Encoder

처리 순서:

1. posed image sequence에서 keyframe을 선택한다.
2. 각 keyframe에서 CNN image feature map을 추출한다.
3. 각 3D point를 keyframe image로 projection한다.
4. projected pixel 위치의 image feature를 가져온다.
5. 여러 view에서 얻은 feature를 평균한다.
6. point의 XYZ 좌표와 image feature를 concatenate한다.
7. enriched point cloud를 sparse 3D conv encoder에 입력한다.

이 방식은 geometry와 semantic/image cue를 함께 사용하므로, 로봇청소기 기반 실내 layout 생성에 가장 현실적인 encoder 후보이다.

##### C. Posed Image Set Encoder

- RayTran과 유사한 2D ↔ 3D bidirectional transformer encoder를 사용한다.
- dense voxel grid feature와 image feature가 patch-voxel ray intersection 기반 attention으로 상호작용한다.
- precomputed point cloud 없이 posed image sequence에서 scene feature를 end-to-end로 만들 수 있다.
- 구현 복잡도는 높다.

---

#### 4) Transformer Language Decoder

Decoder는 Transformer decoder이다.

- 입력:
  - encoder가 만든 latent scene code
  - 이전까지 생성된 SceneScript token sequence
- 출력:
  - 다음 token

학습 시에는 teacher forcing을 사용하고, loss는 token-level cross entropy이다.

```text
Loss = Σ_t CE(pred_token_t, gt_token_t)
```

추론 시에는 다음 과정을 반복한다.

```text
tokens = [START]

while last_token != STOP:
    next_token = decoder(scene_code, tokens)
    tokens.append(next_token)
```

즉, GPT처럼 autoregressive하게 3D scene command token을 생성한다.

---

#### 5) Tokenization

구조화 command를 Transformer가 예측 가능한 integer token sequence로 바꾼다.

기본 schema는 다음과 같다.

```text
[
  START,
  PART,
  CMD,
  PARAM_1,
  PARAM_2,
  ...,
  PARAM_N,
  PART,
  ...,
  STOP
]
```

연속 좌표와 크기 parameter는 **5cm resolution**으로 discretize한다.

예:

```text
x = 1.23m
resolution = 0.05m
token = round(1.23 / 0.05) = 25
```

따라서 continuous regression이 아니라 discrete token classification 문제로 변환된다.

---

#### 6) Command 확장

Object를 bbox보다 자세히 표현하기 위해 `make_prim` command를 추가한다.

```text
make_prim:
    bbox_id,
    prim_num,
    class,
    center_x,
    center_y,
    center_z,
    angle_x,
    angle_y,
    angle_z,
    scale_x,
    scale_y,
    scale_z
```

이 command는 object part를 cuboid 또는 extruded cylinder 같은 volumetric primitive로 표현한다.

예:

```text
make_bbox(class="table", ...)
make_prim(bbox_id=0, prim_num=0, class="cuboid", part="tabletop", ...)
make_prim(bbox_id=0, prim_num=1, class="cuboid", part="leg", ...)
```

---

#### 7) Evaluation

Layout entity는 4개 corner를 가진 3D plane segment로 평가한다.

```text
E = {c1, c2, c3, c4}
```

예측 entity와 GT entity 사이의 거리는 Hungarian matching으로 corner correspondence를 맞춘 뒤, 가장 큰 corner distance로 정의한다.

```text
dE(E, E') = max ||ci - c'π(i)||
```

이 distance가 threshold 이하이면 correct prediction으로 본다.  
논문은 여러 threshold에서 F1 score를 계산하고 평균낸다.

Object detection의 경우 SceneScript는 `make_bbox`별 confidence score를 예측하지 않기 때문에, 일반 mAP 대신 F1 기반 metric을 사용한다.

---

### 출력

- **SceneScript command sequence**
  - `make_wall(...)`
  - `make_door(...)`
  - `make_window(...)`
  - `make_bbox(...)`
  - optional: `make_prim(...)`

- **3D architectural layout**
  - 벽
  - 문
  - 창문
  - 방 구조

- **3D object detection 결과**
  - 객체 class
  - 3D center
  - yaw
  - 3D size
  - oriented 3D bounding box

- **Coarse object reconstruction**
  - table/chair/sofa 등을 cuboid 또는 extruded cylinder primitive로 표현

- **Interpreter를 통한 editable 3D scene**
  - JSON, GLB, CAD-like script, floorplan 등으로 변환 가능

---

## 4. 메모

- SceneScript는 **dense reconstruction 모델**이라기보다 **scene-to-program generation 모델**에 가깝다.
- 로봇청소기 스캔처럼 시야가 낮고 occlusion이 많은 환경에서는 정밀 mesh보다 **parametric layout + 3D bbox + coarse primitive**가 더 현실적인 목표이다.
- 사용자 목적에 맞게 적용한다면 다음 구조가 적합하다.

```text
로봇청소기 RGB-D / LiDAR / SLAM
    ↓
pose graph + global point cloud
    ↓
floor alignment + 5cm voxelization
    ↓
sparse 3D conv point encoder
    ↓
RGB lifted feature 추가
    ↓
Transformer command decoder
    ↓
make_wall / make_door / make_bbox sequence
    ↓
3D house layout JSON / GLB / floorplan
```

- 실제 로봇청소기 적용 시 추가로 필요한 요소:
  - floor plane 기준 좌표계 정렬
  - gravity alignment
  - wall snapping
  - room polygon closure
  - door/window-wall relation validation
  - bbox confidence 추가
  - overlapping object merge
  - occlusion-aware wall/object completion
  - low-height robot viewpoint에 맞춘 synthetic data augmentation

- 기본 SceneScript에는 confidence score가 없기 때문에, 실서비스용 3D object detection에서는 `make_bbox`에 confidence parameter를 추가하는 것이 좋다.

```text
make_bbox:
    id,
    class,
    confidence,
    position_x,
    position_y,
    position_z,
    angle_z,
    scale_x,
    scale_y,
    scale_z
```

- SceneScript의 가장 큰 장점은 command를 추가하는 것만으로 표현을 확장할 수 있다는 점이다.
  - `make_room(...)`
  - `make_floor(...)`
  - `make_ceiling(...)`
  - `make_cabinet(...)`
  - `make_appliance(...)`
  - `make_state(door_open_degree=...)`
  - `make_support_relation(object_id, support_id)`

- 한계:
  - command가 수동 설계되어야 한다.
  - mm-level fine geometry 복원에는 적합하지 않다.
  - 결과가 command grammar에 의존한다.
  - 학습을 위해 많은 synthetic data가 필요하다.
  - Project Aria 입력을 로봇청소기 센서 입력으로 바꾸려면 domain adaptation이 필요하다.

---

## 5. 적용 포인트
 - Aria synthetic dataset을 3d parametric reconstruction 모델 학습에 활용 가능
 - 3d parametric reconstruction baseline으로 검토할 만함
