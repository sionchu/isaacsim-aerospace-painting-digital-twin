# Project handoff

## Objective

Extend the portable aerospace painting demo toward a multi-fidelity generic
air-assisted spray workflow without overstating geometric coverage as paint
physics.

## Completed checkpoint

- Feature branch: `feat/air-assisted-spray-physics`
- Baseline geometric model is preserved.
- Added the air-assisted parameter schema, nozzle-frame contract, analytic
  anisotropic footprint baseline, and mass-ledger utility.
- Added solver decision, licensing, research, architecture, and fidelity docs.
- Portable tests pass.
- Installed OpenCFD/Keysight OpenFOAM v2606 in Ubuntu-24.04 WSL2 from the
  official `dl.openfoam.com` Debian repository.
- Executed the stationary flat-plate `sprayFoam` reference case at coarse
  (6,144 cells) and medium (27,648 cells) resolution using five configured
  ConeNozzleInjection bins and localInteraction wall stick/escape handling.
- Wrote v2606 mass-ledger JSON, deposition NPZ/CSV maps, carrier-air velocity
  visualizations, and mesh-sensitivity outputs under `results/` and `media/`.
- Added the medium 15-degree incidence input set, changing only the air-inlet
  vector and five injection directions, and verified that strict input audit.
- Executed the real incidence `sprayFoam` case at 27,648 cells.  `blockMesh`,
  `checkMesh` (Mesh OK), and `sprayFoam` all completed with the solver `End`
  marker; the final wall ledger records 9.69281800814e-7 kg stuck and
  3.0718199183e-8 kg escaped.
- Added covariance/eigen footprint metrics, incidence response/noise JSON,
  deposition maps, comparison media, and OpenFOAM-free regression tests.
- Added the S2 covariance-moment surrogate using the existing
  `AnisotropicGaussian` renderer.  The 0-degree and 15-degree medium teacher
  reconstructions pass the single-Gaussian warning bands.
- Froze the 7.5-degree prediction before CFD in
  `results/air_assisted/s2/holdout_7p5_prediction.json` with canonical hash
  `30890457e2e9be65d18c05719119b29feb9311f6b8e47926cc366badf88dfba7`.
- Executed the new 7.5-degree medium OpenFOAM case at 27,648 cells.  The
  solver reached `End`, `checkMesh` reported `Mesh OK`, and the target wall
  deposited `1.000000000007e-6 kg` with a closed ledger.
- S2 hold-out validation passes: centroid distance `4.0323913e-4 m`, sigma
  major relative error `4.8839%`, sigma minor relative error `1.2748%`, field
  NRMSE `0.03316`, and correlation `0.93065`; S2 improves over S1.
- Repository verification: 30 tests passed, Python compilation passed, and
  `git diff --check` passed.

## Current level

`S2 + native runtime` — the CFD-calibrated covariance-moment deposition
surrogate is validated on a held-out 7.5-degree OpenFOAM v2606 medium case and
has completed the native Isaac Sim integration checkpoint. The calibrated
interval remains `0-15 degrees`; no extrapolation is used.

## Next checkpoint

The next concrete action is an optional separate fidelity task for NVIDIA Warp
GPU Lagrangian transport. Isaac Lab and reinforcement learning remain out of
scope for this checkpoint.

## Native S2 runtime checkpoint (2026-09-08)

### Objective

Integrate the frozen S2 deposition surrogate into the native Windows Isaac Sim
painting cell using the existing robot/tool motion, curved-surface accumulator,
live density overlay, and reviewed runtime evidence.

### Scope

- Native Isaac Sim 6.0.1 only; the existing local installation was reused.
- Existing composed painting cell and `PaintingCycle`; no toy scene or detached
  nozzle animation.
- S2 v1 fixed operating point: 0-15 degree fan-major incidence, 0.24 m
  stand-off, 18 m/s air, and 1e-4 kg/s liquid flow.
- One 24 s, 720-frame run at 30 Hz with the 0 degree, 7.5 degree, and 15 degree
  spray sections.

### Acceptance criteria

The native stage must load with the actual robot, carriage, process tool,
spray tool, and panel prims; actual TCP/tool transforms must drive the S2
frame; the curved surface map must use geometric area weights; incidence and
minor-plane validity guards must be active; the mass ledger and surface-map
closure must close; no out-of-domain step may deposit; performance and reviewed
media must be recorded; and portable verification must remain Isaac-free.

### Completed

- The previous native-availability blocker was cleared without reinstalling
  Isaac Sim. The validated branch/base remains
  `feat/air-assisted-spray-physics` / `ae724ac534c89a8d8ee9a3093608e6efcbea878c`.
- Native Isaac Sim version `6.0.1-rc.7+release.42383.32955d8d.gl` executed on
  the RTX 4070 Ti. The selected stage is
  `../visual_prototypes/scenes/aircraft_painting_cell.usda`.
- Actual prims resolved: root `/World/AerospacePaintingCell`, carriage
  `/World/AerospacePaintingCell/LinearTrack/Carriage`, robot
  `/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC`,
  process tool `/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC/Asset/J6_link/flange/ProcessTool`,
  spray tool `/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC/Asset/J6_link/flange/ProcessTool/SprayGun`,
  and workpiece `/World/Aircraft/Panel/Asset`.
- `PaintingCycle` drives the real robot chain and was parameterized for the S2
  0.24 m stand-off. Non-spray step-over intervals rotate the real TCP through
  0 to 7.5 to 15 degrees while preserving IK continuity.
- Added the Isaac-free runtime adapter, deterministic model package, native
  entry point, structured curved-surface accumulator, overlay mesh, metrics,
  and reviewed media. The model hash is
  `ce0a9f2d4adfe13d2fdba5ba6cf85bdf7bf05a2968b03930686938ba82c22f43`.

### Current checkpoint

`ISAAC_S2_RUNTIME_VALIDATED`.

The final native run completed 720 frames with 450 spray updates. Injected
mass was `0.00150208623087621 kg`, deposited mass
`0.00147901553886492 kg`, modeled overspray
`0.0000230706920112933 kg`, ledger residual
`-5.62429876976855e-18 kg`, map closure error
`1.95156391047391e-18 kg`, and zero out-of-domain steps. The run measured
1,159 surface elements, mean S2 update time `0.000217757 s`, p95
`0.000398300 s`, and a 30 Hz simulated overlay update cadence.

### Decisions and reasons

- The existing composed scene and joint-chain motion are the runtime authority;
  the adapter reads the live TCP and SprayGun transforms.
- Incidence transitions run during non-spray step-over intervals so the actual
  robot remains coherent instead of switching wrist branches discontinuously.
- The numeric adapter remains Isaac-free; Isaac-specific work is confined to
  the native entry point and a single bounded overlay mesh.
- Viewport utility capture is used for the evidence sequence because the
  earlier Replicator capture path did not complete reliably in this scene.

### Verification evidence

- Native command: `native Isaac Sim python.bat scripts/run_isaac_s2_runtime.py`.
  The run printed `SCENE_GATE_PASS`, resolved the required prims, printed the
  three process sections, printed `RUNTIME_PASS frames=720`, and shut down the
  Simulation App cleanly.
- Process-state spray measurements: A 0 degrees, B 7.49999999 degrees, and C
  14.99999956 degrees mean incidence; spray stand-off means are
  0.240000000006 m, 0.240000000009 m, and 0.240000000259 m; transfer
  efficiencies are 1.0, 0.984640900419, and 0.969281800814.
- Portable verification: `pytest -q` -> `30 passed`; `python -m compileall -q
  src scripts` passed; `git diff --check` passed.
- Provenance is embedded in `models/s2_air_assisted_v1.json` and checked by the
  runtime loader against its source hashes and validated teacher commit.
- Metrics: `results/air_assisted/isaac_s2_runtime/runtime_metrics.json`.
  Media: `media/air_assisted_spray/isaac_s2_runtime_hero.png`,
  `media/air_assisted_spray/isaac_s2_runtime_deposition.png`, and
  `media/air_assisted_spray/isaac_s2_runtime_demo.mp4`.
- The final video is H.264, 1920x1080, 30 fps, and 24.133333 s. No raw USD,
  Isaac cache, shader cache, temporary frame sequence, or local log is part of
  the validation artifact set.

### Not executed

- NVIDIA Warp droplet transport, Isaac Lab, reinforcement learning, and any
  production coating/film-thickness qualification were not executed.
- No new CFD case was created and no droplet-resolved or primary-atomization
  runtime was claimed.

### Blockers

None for this checkpoint.

### Modified files

- `HANDOFF.md`
- `models/s2_air_assisted_v1.json`
- `scripts/build_s2_model_package.py`
- `scripts/run_isaac_s2_runtime.py`
- `src/aerospace_painting/s2_runtime.py`
- `tests/test_s2_runtime.py`
- `results/air_assisted/isaac_s2_runtime/runtime_metrics.json`
- `media/air_assisted_spray/isaac_s2_runtime_hero.png`
- `media/air_assisted_spray/isaac_s2_runtime_deposition.png`
- `media/air_assisted_spray/isaac_s2_runtime_demo.mp4`
- Shared motion helper updated outside this repository at
  `../visual_prototypes/aerospace_process_motion.py` to expose the S2 stand-off
  and incidence frame while preserving its existing joint motion.

### Next concrete action

If another fidelity layer is authorized, add NVIDIA Warp GPU Lagrangian droplet
transport and compare its deposited footprint with the frozen S2/OpenFOAM
reference. Do not start Isaac Lab yet.
