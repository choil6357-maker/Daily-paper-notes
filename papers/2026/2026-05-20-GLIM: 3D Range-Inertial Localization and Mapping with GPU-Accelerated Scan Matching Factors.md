# GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors

- **학회/저널:** Robotics and Autonomous Systems, Vol. 179, 2024
- **링크:** [arXiv](https://arxiv.org/abs/2407.10344) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0921889024001349)
- **코드:** [GitHub - koide3/glim](https://github.com/koide3/glim) / [Project Page](https://koide3.github.io/glim/)
- **분야:** 3D SLAM, LiDAR-Inertial SLAM, Range-IMU Mapping, GPU-Accelerated Scan Matching, Factor Graph Optimization

---

## 1. 요약

- GLIM은 LiDAR, depth camera, stereo camera 등 다양한 range sensor와 IMU를 결합한 3D localization 및 mapping 프레임워크이다.
- 기존 range-based SLAM에서 많이 쓰이던 `frame-to-model matching + filtering + pose graph optimization` 구조의 한계를 개선한다.
- Odometry 단계에서는 `fixed-lag smoothing`과 `keyframe-based point cloud matching`을 결합하여 최근 몇 초 동안의 sensor state를 계속 재최적화한다.
- 이를 통해 range data가 잠시 degenerate되는 상황, 예를 들어 평평한 벽, 복도, featureless 환경에서도 pose drift를 줄일 수 있다.
- Point cloud matching cost는 GPU 가속 가능한 `voxelized GICP, VGICP` 기반 registration error factor로 구성된다.
- Global mapping에서는 submap 간 relative pose constraint를 Gaussian으로 근사하지 않고, submap 간 registration error 자체를 직접 최소화한다.
- Local mapping은 odometry에서 marginalization된 frame들을 모아 submap으로 만들고, submap 내부 frame들에 대해 all-to-all registration을 수행한다.
- Global mapping은 overlap이 있는 submap pair 사이에 dense matching factor를 생성하여 loop closure를 암묵적으로 수행한다.
- Submap 사이의 IMU 제약을 안정적으로 사용하기 위해 각 submap에 left/right endpoint state를 도입한다.
- 실험에서는 FAST-LIO2, LIO-SAM, VoxelMap, BALM, SLICT 등과 비교하여 range degeneration, 다양한 sensor, UAV dynamic motion 환경에서 높은 정확도와 강인성을 보였다.

---

## 2. 핵심 기여

- **GPU 가속 VGICP matching factor**
  - 기존 voxelized GICP를 확장하여 point cloud registration error를 factor graph 안에서 직접 계산한다.
  - Surface-orientation 기반 correspondence validation을 추가하여 얇은 벽의 앞면/뒷면이 잘못 매칭되는 문제를 줄인다.
  - Multi-resolution voxelmap을 사용하여 다양한 환경 scale에서 안정적인 correspondence를 얻는다.

- **Fixed-lag smoothing 기반 range-IMU odometry**
  - 최신 frame 하나만 filtering으로 추정하는 대신, 최근 몇 초간의 frame들을 active window에 유지하고 계속 재최적화한다.
  - Range data가 순간적으로 degenerate되어도 pose를 바로 확정하지 않고, 이후 충분한 geometric constraint가 들어오면 과거 pose까지 보정한다.
  - Keyframe matching과 직전 frame matching을 함께 사용하여 drift 감소와 빠른 sensor motion 대응을 동시에 수행한다.

- **Global registration error minimization**
  - 기존 pose graph optimization처럼 scan matching 결과를 relative pose Gaussian constraint로 변환하지 않는다.
  - Submap 간 point cloud registration error를 직접 factor graph에서 최소화한다.
  - 작은 overlap을 가진 submap pair도 활용할 수 있어 large loop trajectory에서 더 일관된 mapping이 가능하다.

- **Endpoint 기반 global IMU constraint**
  - Submap 사이에 직접 IMU factor를 넣으면 integration time이 길어져 uncertainty가 커진다.
  - GLIM은 각 submap의 첫 frame과 마지막 frame을 endpoint로 두고, 연속 submap의 endpoint 사이에 IMU factor를 생성한다.
  - 이를 통해 짧은 scan interval에 해당하는 강한 IMU constraint를 global optimization에 유지할 수 있다.

- **다양한 range sensor 지원**
  - Spinning LiDAR, non-repetitive LiDAR, ToF depth camera, solid-state LiDAR, active/passive stereo camera 등 다양한 sensor에서 동작 가능함을 보였다.
  - Feature extraction에 의존하지 않는 direct distribution-to-distribution matching을 사용하기 때문에 noisy point cloud에도 비교적 강인하다.

---

## 3. 방법

### 입력

- **Range sensor point cloud**
  - LiDAR
  - Depth camera
  - Stereo camera
  - Solid-state LiDAR 등

- **IMU measurements**
  - Linear acceleration
  - Angular velocity

- **Optional camera images**
  - Multi-camera visual feature constraint를 위한 이미지 입력

- **Calibration 정보**
  - Range sensor, IMU, camera 사이의 extrinsic transformation은 알고 있다고 가정한다.
  - 내부 처리에서는 point cloud를 IMU frame으로 변환하여 통합 sensor coordinate에서 다룬다.

---

### 핵심 아이디어

#### 1. 전체 구조

GLIM은 다음 네 단계로 구성된다.

```text
Input range-IMU data
        ↓
Preprocessing
        ↓
Odometry Estimation
        ↓
Local Mapping
        ↓
Global Mapping
        ↓
Globally consistent trajectory and map
```

각 단계는 모두 factor graph 기반이며, range constraint와 IMU constraint를 tight coupling 방식으로 사용한다.

---

#### 2. Preprocessing

- 입력 point cloud를 downsample한다.
- 각 point에 대해 k-nearest neighbor를 미리 계산한다.
- IMU motion prediction을 이용해 point cloud distortion을 deskewing한다.
- 각 point 주변의 local surface shape을 covariance로 표현한다.

각 point는 다음과 같은 Gaussian distribution으로 표현된다.

```text
p_k = (μ_k, C_k)
```

여기서 `μ_k`는 point 위치이고, `C_k`는 주변 point들로부터 계산한 covariance이다.

---

#### 3. Matching Cost Factor

GLIM의 point cloud matching은 VGICP 기반이다.

기본 개념은 다음과 같다.

```text
point distribution ↔ voxel distribution
```

즉, 단순 point-to-point 또는 point-to-plane matching이 아니라, covariance를 가진 distribution-to-distribution matching을 수행한다.

또한 multi-resolution voxelmap을 사용한다.

```text
r0
2r0
4r0
...
```

이를 통해 coarse-to-fine matching 효과를 얻고, 초기 pose 오차가 있어도 더 안정적으로 수렴할 수 있다.

실내 환경에서는 얇은 벽의 양쪽 면이 모두 관측될 수 있으므로, surface normal을 이용해 상대 viewpoint에서 실제로 보이는 point인지 검사한다. 보이지 않는 point는 correspondence에서 제거한다.

---

#### 4. Odometry Estimation

기존 방식은 보통 다음과 같다.

```text
current scan → fixed local map에 matching
latest state만 filtering으로 update
```

GLIM은 다음과 같이 동작한다.

```text
recent active frames + keyframes
        ↓
fixed-lag smoothing으로 계속 재최적화
```

Odometry graph에는 다음 factor들이 들어간다.

```text
1. latest frame ↔ previous frames matching factor
2. latest frame ↔ keyframes matching factor
3. consecutive frames 사이 IMU preintegration factor
4. marginalization prior
```

직전 frame들과의 matching은 빠른 sensor motion에 대응하기 위한 것이고, keyframe matching은 drift를 줄이기 위한 것이다.

Fixed-lag smoothing을 사용하기 때문에 최근 몇 초 동안의 frame들은 active 상태로 유지된다. Range data가 잠시 degenerate되어도 pose를 바로 확정하지 않고, 이후 좋은 geometric constraint가 들어오면 과거 pose까지 다시 보정할 수 있다.

---

#### 5. Optional Multi-Camera Visual Constraint

GLIM은 multi-camera visual constraint를 추가할 수 있다.

- 각 camera에서 2D visual feature를 검출하고 tracking한다.
- RANSAC essential matrix estimation으로 outlier를 제거한다.
- LiDAR-IMU pose를 기반으로 triangulation하고 reprojection error가 큰 feature를 제거한다.
- 통과한 feature는 landmark로 등록되고, projection factor로 factor graph에 들어간다.

최종적으로 다음 constraint들이 하나의 factor graph에서 joint optimization된다.

```text
Range matching factor
+ IMU preintegration factor
+ Visual projection factor
+ Visual keyframe interpolation factor
```

---

#### 6. Local Mapping

Odometry graph에서 marginalization된 frame들은 local mapping module로 전달된다.

Local mapping의 목적은 여러 frame을 하나의 submap으로 정밀하게 묶는 것이다.

Submap 내부에서는 모든 frame 조합에 대해 matching cost factor를 생성한다.

```text
frame 1 ↔ frame 2
frame 1 ↔ frame 3
frame 2 ↔ frame 3
...
```

즉, submap 내부 all-to-all registration을 수행한다.

또한 consecutive frame 사이에는 IMU preintegration factor를 넣고, 각 frame의 velocity와 IMU bias가 odometry에서 얻은 값과 크게 달라지지 않도록 prior factor를 추가한다.

Submap은 다음 조건 중 하나를 만족하면 최적화 후 병합된다.

```text
1. submap 내부 frame 수가 N_sub에 도달
2. 첫 frame과 마지막 frame의 overlap이 threshold보다 작아짐
```

Local mapping은 batch optimization을 수행하므로, odometry의 짧은 fixed-lag window로는 충분히 보정하기 어려운 drift를 submap 내부에서 추가로 줄일 수 있다.

---

#### 7. Global Mapping

Global mapping은 submap pose들을 전체적으로 최적화한다.

기존 pose graph optimization은 보통 다음처럼 동작한다.

```text
scan matching으로 relative pose 계산
        ↓
relative pose를 Gaussian constraint로 pose graph에 삽입
        ↓
pose graph optimization
```

GLIM은 다르게 동작한다.

```text
submap A와 submap B의 registration error 직접 계산
        ↓
그 error 자체를 factor graph에서 최소화
```

Overlap이 작은 threshold 이상인 submap pair 사이에는 matching cost factor를 생성한다.

```text
overlap > threshold
        ↓
submap-to-submap matching factor 생성
```

이렇게 하면 revisit된 장소의 submap끼리 자연스럽게 연결되며, 명시적인 loop closure event 없이도 loop closure 효과가 발생한다.

---

#### 8. Endpoint 기반 IMU Constraint

Submap은 frame보다 긴 시간 간격으로 생성되기 때문에, submap pose 사이에 직접 IMU factor를 넣으면 integration time이 길어져 uncertainty가 커진다.

이를 해결하기 위해 GLIM은 각 submap에 두 개의 endpoint state를 둔다.

```text
x_L^i = submap i의 첫 frame 상태
x_R^i = submap i의 마지막 frame 상태
```

그리고 IMU factor는 다음과 같이 연결한다.

```text
submap i의 right endpoint
        ↕
IMU factor
        ↕
submap i+1의 left endpoint
```

이렇게 하면 IMU factor가 짧은 scan interval만 포함하므로 uncertainty가 작고, submap pose를 강하게 제약할 수 있다.

---

### 출력

- **실시간 odometry trajectory**
  - Fixed-lag smoothing 기반으로 추정된 low-drift sensor trajectory

- **Local submaps**
  - 여러 frame을 all-to-all registration으로 정렬한 submap

- **Globally optimized trajectory**
  - Submap 간 registration error minimization과 endpoint IMU constraint를 통해 보정된 global trajectory

- **Consistent 3D map**
  - Loop closure와 global mapping을 거친 전역 일관성 높은 point cloud map

- **Optional visual-range-IMU trajectory**
  - Multi-camera visual constraint를 추가한 경우, visual feature까지 tight coupling한 trajectory

---

## 4. 메모

- GLIM의 핵심은 `pose graph optimization`을 완전히 버린다기보다는, 기존의 relative pose Gaussian constraint 중심 pose graph를 `registration error 직접 최소화 factor graph`로 바꾼 것이다.
- FAST-LIO2류 방법은 frame-to-model matching 기반이기 때문에, range data가 완전히 degenerate되는 순간 pose와 map model이 손상될 수 있다.
- GLIM은 fixed-lag smoothing으로 최근 frame들을 보류하고 계속 재최적화하기 때문에, 이후 geometric constraint가 회복되면 과거 state까지 보정할 수 있다.
- Global mapping에서 dense submap matching factor를 사용하므로, loop closure가 별도의 이벤트라기보다 overlap이 있는 submap pair를 연결하는 방식으로 자연스럽게 수행된다.
- Endpoint mechanism은 global optimization에서 IMU 정보를 제대로 쓰기 위한 중요한 장치이다.
- 실험적으로 GLIM은 Newer College Dataset, NTU VIRAL Dataset, 다양한 range-IMU sensor 실험에서 강인성과 범용성을 보였다.
- 다만 long-term range degeneration은 여전히 어려운 문제로 남아 있다.
- 논문에서는 이를 해결하기 위한 방향으로 camera, radar, wheel odometry 같은 추가 sensor 통합 또는 learning-based motion estimation 통합을 제안한다.
- 사용자가 실내 집 환경 3D reconstruction을 목표로 한다면, GLIM은 LiDAR pose estimation과 mapping backbone으로 유용할 수 있다.
- 하지만 GLIM의 출력은 기본적으로 point cloud map이며, 3D depth completion용 dense GT mesh를 바로 보장하는 방식은 아니다. 후처리로 TSDF fusion, mesh reconstruction, reflective surface filtering, manual/automatic 3D bbox annotation guide와 결합하는 것이 현실적이다.

---

## 5. 적용 포인트
 - LiDAR 기반 SLAM의 baseline
