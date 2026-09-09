# 로봇 스프레이 시각화에서 CFD 기반 전체 패널 도장 Digital Twin으로 확장했습니다

최근 Isaac Sim을 이용한 항공 로봇 도장 초기 프로토타입을 공유했는데, 이번에는
단순히 로봇이 도장 경로를 따라 움직이거나 스프레이 콘을 보여주는 수준을
넘어 확장했습니다.

현재 버전은 다음 네 계층을 연결합니다.

OpenFOAM → S2 deposition model → NVIDIA Warp → Isaac Sim

OpenFOAM은 오프라인 CFD 기준 유동장을 제공합니다.  CFD 결과를 기반으로
보정한 S2 surrogate는 빠른 도포 질량 예측을 담당합니다.  NVIDIA Warp는 GPU
기반 Lagrangian droplet transport를 계산합니다.  Isaac Sim에서는 로봇, 7축
레일, 분사 공정, reference flow 시각화와 전체 패널 결과를 하나의 환경에서
연결했습니다.

현재 데모에서는 2.85 m² 곡면 패널에 대해 28회의 도장 패스를 수행하고, 전체
면에 누적되는 예측 도포 질량을 계산합니다.

이 누적 질량으로부터 wet-film thickness(WFT)도 추정했습니다.

- 평균 Estimated WFT: 3.65 µm
- P05–P95: 2.12–3.90 µm
- CV: 12.6%
- 전체 면적의 94.6%가 예측 평균의 ±20% 범위

Transport 계층에서는 1,722개 Warp batch와 batch당 2,500개 particle을
사용해 총 430.5만 개의 computational parcel history를 계산했습니다.

또 OpenFOAM에서 계산된 실제 reference velocity field를 velocity vector,
magnitude slice, streamline으로 시각화하고, 같은 simulated moment에 actual
Warp particle 위치, 짧은 particle trail, mesh-hit impact marker, progressive
S2 WFT overlay를 함께 표시했습니다.  Technical view에서는 visual-only plume
guide를 끄므로, 강조된 particle·trail·impact는 기록된 Warp 위치와 hit
event에서 만들어집니다.  Warp hit map은 diagnostic layer이며 WFT authority인
S2를 대체하지 않습니다.

최종 technical capture는 85초 1920×1080 H.264 영상으로,
PROCESS FRAME → CFD FLOW → ACTUAL WARP DROPLETS + IMPACTS → FULL-PANEL
COATING BUILD-UP → ESTIMATED WFT 순서로 구성했습니다.

중요한 기술적 경계가 있습니다.  OpenFOAM CFD는 오프라인에서 계산한 뒤
Isaac Sim의 로컬 공정 좌표계에 매핑하여 시각화합니다.  Isaac Sim 내부에서
CFD를 실시간으로 풀고 있는 것은 아닙니다.  Moving Warp integration은
quasi-steady local tangent-patch approximation입니다.  WFT 역시 누적 도포
질량과 설정된 액체 밀도로 계산한 engineering estimate이며, 실제 측정값이나
생산 공정 인증 수준의 도막 두께 예측이 아닙니다.

현재 모델은 primary atomization, breakup, evaporation, stochastic turbulent
dispersion, splash/rebound, wall-film transport, sagging, curing, measured
dry-film thickness, production coating qualification을 검증하지 않습니다.

이번 프로젝트에서 가장 흥미로운 흐름은 다음과 같습니다.

CFD reference → fast surrogate → GPU transport → robot/process digital twin →
surface result

#IsaacSim #NVIDIAWarp #OpenFOAM #DigitalTwin #Robotics #IndustrialRobotics #PhysicalAI #Simulation #Manufacturing #Aerospace #CFD
