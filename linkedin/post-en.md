# Beyond Offline Programming: Exploring an Aerospace Robotic Painting Digital Twin with Isaac Sim, OpenFOAM, and NVIDIA Warp

Mature OLP platforms such as RoboDK and DELMIA already provide strong
workflows for CAD-based path generation, robot simulation, offline programming,
and production integration.  This project does not try to replace them.

Instead, I explored a complementary question: how can robot motion, process
evaluation, and GPU spray transport share one OpenUSD/Isaac Sim scene?

The release candidate connects:

- a rail-mounted 6-axis robot and actual TCP-driven motion;
- a generic aircraft-panel workpiece and process tool;
- OpenFOAM v2606 offline reference cases;
- an S2 CFD-calibrated deposition surrogate validated on a 7.5° hold-out; and
- a validated W1.3 carrier model with an optional native Warp plume layer.

The native viewer makes the distinction visible: S2 remains the authoritative
surface deposition overlay, while Warp shows GPU Lagrangian transport with
drag, gravity, and mesh collision.  The moving-scene transport is a
quasi-steady local tangent-patch approximation.

This is not a production paint-quality solver.  It does not claim primary
atomization, breakup, evaporation, splash/rebound, wall-film transport,
curing, physical film thickness, or production coating qualification.

The repository contains the portable math, OpenFOAM reference evidence,
runtime metrics, architecture, scope boundaries, and reviewed demo media.

#NVIDIAIsaacSim #OpenFOAM #NVIDIAWarp #OpenUSD #DigitalTwin #Robotics #AerospaceManufacturing
