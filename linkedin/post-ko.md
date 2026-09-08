# 기존 OLP를 넘어, Isaac Sim·OpenFOAM·NVIDIA Warp로 연결한 항공기 로봇 도장 Digital Twin

RoboDK와 DELMIA 같은 성숙한 OLP 도구는 CAD 기반 경로 생성, reachability,
충돌 검토, 오프라인 프로그래밍과 생산 연계를 이미 잘 제공합니다. 이
프로젝트는 그 도구들을 대체하려는 시도가 아닙니다.

로봇 motion, 공정 평가, CFD reference, GPU spray transport를 하나의
OpenUSD/Isaac Sim 장면 안에서 연결하면 어떤 공정 시야를 만들 수 있는지
살펴보는 complementary portfolio입니다.

이번 release candidate에는 다음이 연결되어 있습니다.

- 실제 TCP motion을 구동하는 rail-mounted 6축 robot
- generic aircraft-panel workpiece와 process tool
- OpenFOAM v2606 offline reference cases
- 7.5° hold-out으로 검증한 S2 CFD-calibrated deposition surrogate
- blind 10° hold-out으로 검증한 W1.3 carrier와 native Warp plume layer

Native viewer에서 역할도 분리해 보입니다. S2가 authoritative surface
deposition overlay를 담당하고, Warp는 drag·gravity·mesh collision을
포함한 GPU Lagrangian transport를 보여줍니다. Moving-scene transport는
quasi-steady local tangent-patch approximation입니다.

이 결과는 production paint-quality solver가 아닙니다. Primary atomization,
breakup, evaporation, splash/rebound, wall-film transport, curing, 실제 film
thickness, production coating qualification을 주장하지 않습니다.

재현 가능한 계산 코드, OpenFOAM reference evidence, runtime metrics,
architecture, scope boundary와 reviewed media를 정리했습니다.

#NVIDIAIsaacSim #OpenFOAM #NVIDIAWarp #OpenUSD #DigitalTwin #Robotics #AerospaceManufacturing
