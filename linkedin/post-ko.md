# 기존 OLP를 넘어, Isaac Sim으로 항공기 로봇 도장 Digital Twin을 구현해봤습니다

로봇 도장 경로 생성과 생산 프로그램 작성에는 이미 RoboDK, DELMIA 같은
성숙한 OLP 도구가 있습니다. 이번 프로젝트는 이를 대체하려는 시도가
아닙니다. 다른 질문에서 시작했습니다.

로봇과 외부축, 곡면 형상, 공정 경로, 센서와 평가 데이터를 하나의
OpenUSD 장면 안에 표현한다면 도장 공정을 어디까지 확장할 수 있을까?

NVIDIA Isaac Sim에서 다음 요소를 구현했습니다.

- 6축 산업용 로봇과 보조 선형축
- generic 항공기 곡면과 고정 치구
- 표면 위치와 normal 샘플링
- normal 방향을 반영한 tool orientation
- 일정 stand-off를 기준으로 한 왕복 sweep 경로
- spray cone의 기하학적 시각화
- 거리와 입사각을 이용한 coverage 누적 및 overlay

이 결과는 실제 도막 품질 해석이 아닙니다. Atomization, droplet deposition,
airflow CFD, curing, 실제 film thickness는 모델링하거나 검증하지 않았습니다.
현재 구현은 로봇 경로와 공정 기하, coverage evaluation을 하나의 Digital
Twin 장면으로 연결한 기술 포트폴리오입니다.

향후에는 실제 측정 데이터와 센서 관측을 연결해 보정과 폐루프 평가로
확장할 수 있습니다. 우선 공개 저장소에는 재현 가능한 계산 코드,
아키텍처, 범위 경계와 데모 미디어를 정리했습니다.

GitHub와 데모 링크는 첫 댓글에 남깁니다.

#NVIDIAIsaacSim #OpenUSD #DigitalTwin #Robotics #AerospaceManufacturing
