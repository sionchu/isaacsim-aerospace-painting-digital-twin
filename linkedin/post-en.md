# Beyond Offline Programming: Exploring an Aerospace Robotic Painting Digital Twin with NVIDIA Isaac Sim

Mature OLP platforms such as RoboDK and DELMIA already provide strong workflows
for CAD-based path generation, robot simulation, offline programming, and
production integration. This project does not try to replace them.

Instead, I explored a complementary question: how can a robotic surface process
be represented in an OpenUSD/Isaac Sim scene where robot motion, an auxiliary
axis, geometry, sensors, process metrics, and future Physical AI workflows can
share the same simulation environment?

The prototype includes:

- a 6-axis industrial robot on an auxiliary linear axis;
- a generic curved aircraft surface and holding fixture;
- surface position and normal sampling;
- surface-normal-aware tool orientation;
- a constant-stand-off boustrophedon path concept;
- spray-cone geometry visualization; and
- distance/angle-based geometric coverage accumulation and overlay.

This is not a physical paint-quality solver. It does not model or validate
atomization, droplet deposition, airflow CFD, curing, or real film thickness.
The current result is a technical portfolio implementation of robot motion,
process geometry, and coverage evaluation inside a shared digital-twin scene.

The repository contains the portable math, architecture, scope boundaries, and
reviewed demo media. GitHub and demo links are in the first comment.

#NVIDIAIsaacSim #OpenUSD #DigitalTwin #Robotics #AerospaceManufacturing
