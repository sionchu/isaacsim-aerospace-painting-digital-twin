# Relationship to mature offline-programming tools

This project is complementary to mature OLP software, not a replacement for
RoboDK, DELMIA, or a production robot-programming system.

## Different strengths

Traditional OLP tools are strong at CAD-to-path generation, reachability,
collision checking, cycle-time analysis, controller-specific post-processing,
and enterprise manufacturing-process integration.

This portfolio focuses on a shared programmable OpenUSD/Isaac Sim scene in
which robot motion, auxiliary axes, geometry, process overlays, CFD-derived
evaluation, sensors, and GPU droplet transport can be inspected together.

## Process-model contribution

The current release adds an offline OpenFOAM v2606 reference layer, an S2
CFD-calibrated deposition surrogate, and an optional Warp Lagrangian plume
layer.  Those layers complement OLP path and reachability workflows by making
process response visible in the same scene; they do not provide controller
post-processors or replace an OLP system.

## Current limitations

- The reference and surrogate layers do not model primary atomization,
  breakup, evaporation, splash/rebound, wall-film transport, curing, physical
  film thickness, or production coating quality.
- The native Warp plume is a quasi-steady local tangent-patch transport
  approximation, not online CFD.
- Production robot post-processing, calibrated cell commissioning, safety
  analysis, and manufacturing certification remain outside this project.
- Any industrial deployment would require domain-specific tooling and
  independent process evidence.

## References

- [RoboDK robot manufacturing and curve-follow workflow](https://robodk.com/doc/en/Robot-Machining.html)
- [RoboDK post-processors](https://robodk.com/doc/en/Post-Processors.html)
- [DELMIA Robotics](https://www.3ds.com/products/delmia/industrial-engineering/robotics)
- [NVIDIA Isaac Sim sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/index.html)
- [NVIDIA Isaac Sim ROS 2 integration](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/ros2_landing_page.html)
