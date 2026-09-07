# Relationship to mature offline-programming tools

This project is not a RoboDK or DELMIA replacement. It explores a complementary
OpenUSD/Isaac Sim representation for a surface-process workflow.

## Where mature OLP tools are strong

RoboDK provides established CAD/curve-follow workflows, robot simulation,
offline programming, controller-specific post-processors, collision checking,
and support for synchronized external axes. DELMIA Robotics integrates product,
process, workcell, robot programming, validation, and broader manufacturing
planning within the 3DEXPERIENCE platform.

Those capabilities are central when the goal is production robot programming,
controller output, mature reachability and cycle-time workflows, or enterprise
manufacturing-process integration.

## Why Isaac Sim/OpenUSD is interesting here

The portfolio investigates a shared programmable scene in which robot motion,
an auxiliary axis, environment geometry, process overlays, sensors, and custom
metrics can coexist. Isaac Sim also provides Python APIs, RTX/physics sensor
simulation, ROS 2 integration, and extension points for synthetic-data and
future robot-learning workflows.

## Current limitations

- Isaac Sim does not provide a built-in aircraft paint-process solver equivalent
  to mature dedicated OLP or process-engineering tools.
- Paint atomization, airflow, deposition, curing, and physical film thickness
  are outside this implementation.
- Production robot post-processing is not the core purpose of this project.
- Custom path and process logic must be engineered, calibrated, and validated.
- Manufacturing process certification requires domain-specific tooling and
  evidence beyond this demo.

## References

- [RoboDK robot manufacturing and curve-follow workflow](https://robodk.com/doc/en/Robot-Machining.html)
- [RoboDK post-processors](https://robodk.com/doc/en/Post-Processors.html)
- [DELMIA Robotics](https://www.3ds.com/products/delmia/industrial-engineering/robotics)
- [NVIDIA Isaac Sim sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/index.html)
- [NVIDIA Isaac Sim ROS 2 integration](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/ros2_landing_page.html)
