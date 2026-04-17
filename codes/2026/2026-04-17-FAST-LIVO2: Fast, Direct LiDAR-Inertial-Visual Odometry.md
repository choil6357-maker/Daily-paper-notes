# 제목: FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry

- 학회: T-RO '24
- 링크: https://arxiv.org/pdf/2408.14035
- 코드: https://github.com/hku-mars/fast-livo2
- 분야: 3D reconstruction

## 1. 요약
- FAST-LIVO2는 **LiDAR, 카메라, IMU**를 하나의 **unified voxel map** 위에서 직접(direct) 방식으로 결합하는 **LiDAR-inertial-visual odometry** 시스템이다.
- 전체 추정은 **ESIKF (Error-State Iterated Kalman Filter)** 로 수행하며, **IMU propagation → LiDAR update → Visual update** 순서의 **sequential update**로 상태를 보정한다.
- LiDAR는 맵의 **기하 구조(평면 복셀)** 를 만들고, visual은 그 위의 일부 LiDAR 포인트를 **visual map point**로 재사용하여 **image patch 기반 sparse-direct alignment**를 수행한다.
- 별도의 visual feature 추출, triangulation, sliding-window optimization에 크게 의존하지 않고, **LiDAR geometry + image photometric error**를 함께 사용해 pose를 정밀하게 맞춘다.
- 로컬 맵은 **Hash table + octree 기반 adaptive voxel map** 으로 관리되며, 평면으로 수렴한 복셀만 유지하여 계산량과 메모리를 제어한다.
- 각 visual map point에는 여러 패치가 붙을 수 있고, 그중 **photometric similarity + viewing angle** 기준으로 가장 좋은 **reference patch**를 선택한다.
- 패치 정렬 정확도를 높이기 위해, LiDAR로 얻은 평면 법선을 초기값으로 사용하고 이후 패치 간 photometric error를 최소화하여 **normal refine**도 수행한다.
- Visual update 전에 현재 FoV 내 visual map point를 복셀 질의와 **on-demand raycasting**으로 찾고, occlusion, depth discontinuity, 큰 시야각 점들을 제거한다.
- LiDAR update는 **point-to-plane residual**, visual update는 **patch photometric residual**을 사용한다.
- 논문의 핵심 목표는 **빠르고 강인하며 픽셀 수준 정밀도를 가진 direct LIVO 시스템**을 만드는 것이다.

## 2. 핵심 기여
- **Sequential-update ESIKF**: LiDAR와 이미지 측정의 차원 및 구조 차이를 해결하기 위해, 같은 시점에서 **LiDAR 먼저, image 나중**으로 순차 업데이트하는 강결합 필터 구조를 제안했다.
- **Unified voxel map 기반 direct fusion**: LiDAR가 만든 평면 복셀 맵을 visual도 그대로 활용하여, **LiDAR point를 visual map point로 재사용**하고 sparse-direct image alignment를 수행한다.
- **정확도 및 강인성 향상 기법**:
  - LiDAR 기반 **plane prior**
  - 더 좋은 **reference patch update**
  - 패치 기반 **normal refine**
  - **on-demand raycasting**
  - **online exposure time estimation**
  등을 통해 visual alignment의 안정성과 정확도를 높였다.

## 3. 방법
- **입력:**
  - 이전 시점의 상태 및 공분산
  - \(t_{k-1} \sim t_k\) 구간의 **IMU 측정값**
  - 현재 카메라 시각 기준으로 재구성된 **LiDAR scan**
  - 현재 **이미지 프레임**
  - **unified voxel map**
    - 평면 복셀
    - visual map point
    - reference patch / patch pyramid
    - plane normal / uncertainty

- **핵심 아이디어:**
  - 카메라 시각 \(t_k\) 를 기준 업데이트 시점으로 잡는다.
  - IMU로 \(t_k\) 까지 **propagation** 하여 prior state를 만든다.
  - LiDAR raw point를 scan recombination 후 맵에 정합하여 **point-to-plane residual**로 먼저 update한다.
  - 그 결과로 맵의 기하 구조를 갱신하고, 평면 복셀 위 일부 점을 visual map point로 관리한다.
  - 현재 FoV에서 보이는 visual map point를 선택하고, reference patch와 현재 이미지 패치 사이의 **photometric error**로 visual update를 수행한다.
  - 이때 patch pyramid, affine warping, plane prior, reference patch update, raycasting, outlier rejection을 사용해 정렬 정확도를 높인다.

- **출력:**
  - 현재 시점의 최종 **pose/state estimate**
    - 자세, 위치, 속도
    - IMU bias
    - 중력
    - exposure-time-related state
  - 해당 추정의 **공분산**
  - 업데이트된 **로컬 voxel map**
    - 성숙한 평면 구조
    - visual map point
    - reference patch / refined normal

## 4. 적용 포인트
- 비선형 kalman filter error-state interated kalman filter를 고려해 볼만하다?
- 속도가 빠른 SLAM이 필요할 때 활용해 볼 수 있는 방법
