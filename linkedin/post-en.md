# From robotic spray visualization to a CFD-referenced full-panel painting digital twin

I recently shared an early Isaac Sim prototype for aerospace robotic painting.
This version pushes it beyond showing a robot path or a spray cone.

The current stack connects four layers:

OpenFOAM → S2 deposition model → NVIDIA Warp → Isaac Sim

OpenFOAM provides the offline CFD reference flow field.  A CFD-calibrated S2
surrogate provides the fast deposited-mass model.  NVIDIA Warp computes
GPU-based Lagrangian droplet transport.  Isaac Sim brings the robot, rail,
spray process, reference flow visualization, and full-panel result into one
environment.

The demo performs 28 painting passes over a 2.85 m² curved generic surface and
accumulates predicted deposited mass across the panel.

From that deposited mass, I estimate wet-film thickness (WFT):

- Mean estimated WFT: 3.65 µm
- P05–P95: 2.12–3.90 µm
- CV: 12.6%
- 94.6% of the panel area is within ±20% of the predicted mean

The transport layer represents 1,722 Warp batches × 2,500 particles per batch,
or 4.305 million computational parcel histories.

The synchronized analysis view now shows the complete chain at one simulated
moment: OpenFOAM reference streamlines and sparse vectors, actual Warp
particle positions, short particle trails, mesh-hit impact markers, and the
progressive S2 WFT overlay.  The technical view disables the visual-only plume
guide, so these highlighted particles, trails, and impacts come from recorded
Warp positions and hit events.  The diagnostic hit map is separate from the
authoritative S2 thickness layer.

The final technical capture is an 85-second 1920×1080 H.264 sequence:
PROCESS FRAME → CFD FLOW → ACTUAL WARP DROPLETS + IMPACTS → FULL-PANEL
COATING BUILD-UP → ESTIMATED WFT.

One important boundary: the OpenFOAM CFD is solved offline and mapped into the
local process frame for visualization.  It is not real-time CFD solved inside
Isaac Sim.  The moving Warp integration is a quasi-steady local tangent-patch
approximation.  WFT is a model-derived engineering estimate based on deposited
mass and configured liquid density, not a measured or production-qualified
coating thickness.

The current model does not validate primary atomization, breakup, evaporation,
stochastic turbulent dispersion, splash/rebound, wall-film transport, sagging,
curing, measured dry-film thickness, or production coating qualification.

For me, the interesting workflow is:

CFD reference → fast surrogate → GPU transport → robot/process digital twin →
surface result

#IsaacSim #NVIDIAWarp #OpenFOAM #DigitalTwin #Robotics #IndustrialRobotics #PhysicalAI #Simulation #Manufacturing #Aerospace #CFD
