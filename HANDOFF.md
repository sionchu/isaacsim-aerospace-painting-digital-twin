# Project handoff

## Objective

Extend the portable aerospace painting demo toward a multi-fidelity generic
air-assisted spray workflow without overstating geometric coverage as paint
physics.

## Current CFD field visualization checkpoint (2026-09-09)

### Result

`ISAAC_CFD_FLOW_VISUALIZATION_VALIDATED` on `feat/cfd-field-visualization`.
The implementation is visualization/integration only.  It does not alter the
OpenFOAM teacher cases, S2 deposition, Warp W1.3 transport, full-panel plan,
WFT calculation, or mass ledgers.

### CFD source and compact artifact

- OpenFOAM v2606 case: `reference_cfd/openfoam_v2606/flat_plate/medium_incidence_7p5deg/`
- Latest time: `0.06`
- Structured field: `24 × 24 × 48 = 27,648` cells, float32 U
- Bounds: `[-0.15, -0.15, 0.0]` to `[0.15, 0.15, 0.24]` m
- Spacing: `(0.0125, 0.0125, 0.005)` m
- U SHA-256: `0e62dfb333728be18533945c5494c778f7466ae5366d2b895e0d0a8945d34357`
- C SHA-256: `d8c4c949c047b2406c5490a8cc2cb75b6c99a609d432024b229eb0973cabecca`
- Artifact: `models/openfoam_flow_field_7p5deg_v1.npz` plus manifest JSON
- Artifact SHA-256: `cc51dc50fb047cce4cd7a54c55603049771a89e29dce45c4daa587f3f4786b02`
- Speed range: `1.110974 / 15.275813 / 22.480917 m/s` (min/mean/max)

The deterministic builder and provenance checks are in
`scripts/build_openfoam_flow_field.py`; the artifact reconstructs the actual
latest-time U/C grid and rejects hash, dimension, bounds, or finite-value
mismatches.

### Isaac Sim flow layer

`scripts/run_isaac_cfd_flow_visualization.py` loads the compact artifact and
maps benchmark `+X → u` fan-major, `+Y → v` travel/fan-minor, and `+Z → w`
inward process direction into the current SprayGun/target tangent frame.  The
native run produced 512 velocity vectors, a 1,152-vertex / 1,081-cell
`y≈-0.006250 m` magnitude slice, and 25 deterministic RK2 streamlines.  The
flow geometry is precomputed once and rigidly transformed per render update.

View toggles are `--view process`, `--view flow`, `--view combined`, and
`--view result`.  The FLOW/combined HUD labels the layer as
`OPENFOAM REFERENCE FLOW — LOCAL TANGENT FRAME` and explicitly states
`Offline CFD → Isaac visualization`.

### Native evidence and performance

- Native command: `Isaac Sim python.bat scripts/run_isaac_cfd_flow_visualization.py --headless --capture-stills --view combined`
- Native result: `CFD_FLOW_PASS vectors=512 slice_vertices=1152 streamlines=25`
- CFD visualization update: mean `0.032908 s`, p95 `0.091538 s`
- Warp update: mean `0.064265 s`, p95 `0.087002 s`
- S2 reference: mean `0.000270 s`, p95 `0.000375 s` (validated full-panel metrics; no S2 step in visual-only capture)
- Combined capture wall time: `13.6580207 s`

Media:

- `media/air_assisted_spray/isaac_cfd_flow_field.png`
- `media/air_assisted_spray/isaac_cfd_warp_combined.png`
- `media/air_assisted_spray/isaac_cfd_flow_slice.png`
- `media/air_assisted_spray/isaac_full_panel_cfd_flow_demo.mp4`

The final demo is H.264, 1920×1080, 30 fps, 70.0 s.  It reuses the validated
full-panel process segment; no CFD or painting rerun was used to assemble it.

The CFD field is solved offline in OpenFOAM and mapped into the current local
tangent process frame in Isaac Sim for visualization.  It is not solved live
inside Isaac Sim.

### Verification

- `python -m pytest -q` → `55 passed, 2 skipped`
- `python -m compileall -q src scripts tests` → passed
- `git diff --check` → passed
- Native stills were opened and inspected: process, flow, combined, slice, and result captures exist at 1920×1080.
- Public safety scan → no machine-specific paths or credential material in the public docs/artifact manifests; no machine-specific USD is part of the artifact set.

Verification and public safety scan are complete.  Commit
`5807e7352d7cce979cafc60fe8fa0580ef976361` is pushed to
`feat/cfd-field-visualization`.  Next action: open the PR to `main`, review the
README/media diff once, and merge only after review; do not merge automatically.

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

`FULL_PANEL_PORTFOLIO_READY` — the full-panel native process is the public
release path.  S2 remains the authoritative deposition layer, W1.3 supplies
the validated fixed-grid carrier, and the native Warp integration provides
quasi-steady local tangent-patch transport in the Isaac Sim scene.

## Next checkpoint

`PR → main review → merge → publish`.  No new CFD, Isaac Lab,
reinforcement-learning, or production coating work is implied by this
checkpoint.

The checkpoint sections below retain the validation journey as historical
evidence.  The full-panel release section at the end is the current release
path.

## Current full-panel release checkpoint (2026-09-09)

### Result

`FULL_PANEL_PORTFOLIO_READY`.  The committed native run remains
`FULL_PANEL_FILM_ESTIMATION_VALIDATED`; this release fix changes reporting and
public wording only.  No native Isaac rerun was performed.

### Corrected Warp reporting

The reporting artifact now keeps the reconstructable runtime fields together:

- `batch_count = 1722`
- `particles_per_batch = 2500`
- `particle_histories = 4305000`
- `warp_batch_count = 1722`
- `warp_particles_per_batch = 2500`
- `warp_particle_histories = 4305000`

The derivation is `closed_batch_count * particle_count_per_batch`, taken from
the committed native runtime ledger.  The source artifact is
`results/air_assisted/full_panel_film/full_panel_metrics.json`; the generator
now computes the same fields from `warp_runtime.as_dict()`.

### Public visual disclosure

The actual live Warp positions are rendered at
`/World/AerospacePaintingCell/WarpSprayPlume`.  The separate
`/World/AerospacePaintingCell/WarpSprayPlumeGuide` is visual-only with
`adds_mass = false`, `adds_deposition = false`, and `visual_only = true`.
The public wording states that the guide does not affect transport,
deposition, forces, or mass accounting, so the visible orange plume is not
described as all physical Warp parcels.

### Public process result

- Surface area: `2.846638288 m²`
- Passes: `28`
- Estimated WFT mean: `3.649458 µm`
- Estimated WFT P05–P95: `2.120723–3.903405 µm`
- Estimated WFT maximum: `3.907579 µm`
- Estimated WFT standard deviation: `0.459557 µm`
- Estimated WFT CV: `12.5925%`
- Area within ±20% of predicted mean: `94.5946%`
- Full-panel combined mass residual: `-8.8839526e-16 kg`

Estimated WFT is `deposited mass / (surface area × configured liquid density)`
and is model-derived, not measured.  DFT remains secondary and is labeled
`Illustrative DFT estimate — assumed 50% volume solids; synthetic_demo_only`.

### Verification evidence

- `python -m pytest -q` → `49 passed, 2 skipped`
- `python -m compileall -q src scripts tests` → passed
- `git diff --check` → passed
- Reporting arithmetic → `1722 × 2500 = 4305000`
- Public safety scan → no machine-specific paths, credential material, or uncommitted USD

### Release media

- Hero: `media/air_assisted_spray/isaac_full_panel_film_hero.png`
- Demo: `media/air_assisted_spray/isaac_full_panel_film_demo.mp4`
- Final: `media/air_assisted_spray/isaac_full_panel_film_final.png`

The former W2 media remain supporting and historical evidence.

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
  the RTX 4070 Ti. The selected stage is the composed aircraft painting-cell
  scene.
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
- The shared motion helper exposes the S2 stand-off and incidence frame while
  preserving its existing joint motion.

### Next concrete action

If another fidelity layer is authorized, continue the static Warp transport
investigation from the W1 evidence below. Do not start Isaac Lab or integrate
Warp into the moving Isaac scene yet.

## NVIDIA Warp W1 static transport checkpoint (2026-09-08)

### Objective

Evaluate a deterministic CUDA Lagrangian spray transport implementation in the
existing NVIDIA Warp runtime against the frozen OpenFOAM v2606 teacher and the
frozen S2 7.5-degree hold-out, without changing the native moving runtime.

### Scope

- Existing Warp 1.13.0 from Isaac Sim 6.0.1 on `cuda:0`; no package upgrade.
- Existing solved canonical OpenFOAM cases only: 0°, 7.5°, and 15° at 0.06 s.
- Five teacher bins, teacher cone/direction settings, Schiller-Naumann sphere
  drag, gravity, mesh ray collision, stick/escape fate handling, and CUDA
  atomic areal-mass accumulation.
- Carrier model fitted from solved 0°/15° `C`/`U` fields only; the 7.5°
  deposition map remained a transport hold-out.

### Acceptance criteria

The CUDA runtime must close relative mass balance to `1e-5`; the 7.5° hold-out
must meet deposited-mass error `<=10%`, centroid error `<=10 mm`, both sigma
errors `<=15%`, field correlation `>=0.80`, field NRMSE `<=0.40`, and numerical
sensitivity below the physical 0°→15° response.

### Completed

- Added `warp_air_field.py` and `warp_spray.py` with deterministic stratified
  five-bin injection, explicit state arrays, explicit Euler stepping,
  Schiller-Naumann drag, gravity, Warp `mesh_query_ray`, and atomic map writes.
- Added `scripts/validate_warp_spray.py` and portable Warp contract tests.
- Verified the existing native runtime reports Warp `1.13.0`, `cuda:0`, one
  CUDA device, and an RTX 4070 Ti; no install or upgrade was performed.
- Reused the existing solved teacher outputs; no new CFD case or moving Isaac
  integration was created.
- Final canonical sensitivity run used 500 low and 2,000 high particles per
  bin (2,500/10,000 total particles), 1,200 substeps at `5e-5 s` over `0.06 s`.

### Current checkpoint

`WARP_TRANSPORT_NOT_VALIDATED`.

Mass balance and sensitivity pass. The 7.5° high ensemble has deposited-mass
relative error `4.13245e-7`, centroid error `8.41450 mm`, major sigma error
`15.6738%`, minor sigma error `3.87015%`, field correlation `0.835503`, and
field NRMSE `0.0628685`. The major sigma is therefore just outside the `15%`
acceptance gate; this is not promoted to a transport validation.

### Decisions and reasons

- The W1 air model is an axisymmetric Gaussian external jet in the nozzle-local
  frame, with +X major/u, +Y minor/v, and +Z spray/stand-off. Its fit targets
  are solved carrier velocities only.
- The sampler follows the installed v2606 `ConeNozzleInjection` behavior:
  uniform cone angle and the implementation's radius/azimuth correlation.
- The frozen S2 model remains a comparison target only. No drag or air-field
  parameter was tuned against the 7.5° deposition map.
- The canonical high count remains 2,000/bin because low/high moments and mass
  are stable; the physical gate failure is retained rather than hidden by
  increasing particle count.

### Verification evidence

- Native probe through the Isaac Sim launcher reported `WARP_VERSION 1.13.0`,
  devices `['cpu', 'cuda:0']`, `DEFAULT_DEVICE cuda:0`, and `CUDA_COUNT 1`.
- CUDA validation command: `scripts/validate_warp_spray.py
  --low-particles-per-bin 500 --high-particles-per-bin 2000` through the Isaac
  Sim launcher.
- The run executed all six low/high angle cases, printed the Warp CUDA device,
  and produced zero mass residual for every case. The 7.5° high case measured
  mean step `0.000265265 s`, p95 `0.000743080 s`, wall `0.320110 s`, 1,200
  kernel launches, and 10,000 particles.
- Portable verification: `pytest -q tests/test_warp_spray.py
  tests/test_spray_frames.py tests/test_mass_accounting.py` -> `8 passed`;
  `python -m compileall -q src/aerospace_painting/warp_air_field.py
  src/aerospace_painting/warp_spray.py scripts/validate_warp_spray.py` passed.
- JSON evidence: `results/air_assisted/warp_w1/environment.json`,
  `particle_sensitivity.json`, `case_0deg_metrics.json`,
  `case_7p5deg_metrics.json`, `case_15deg_metrics.json`, and
  `validation_summary.json`.
- Media evidence: `media/air_assisted_spray/warp_w1_air_field_fit.png`,
  `warp_w1_openfoam_7p5_comparison.png`, and `warp_w1_angle_response.png`.

### Not executed

- No new OpenFOAM case, deposition DOE, Isaac Sim moving-scene Warp
  integration, Isaac Lab, reinforcement learning, or production coating
  qualification was executed.

### Blockers

The remaining blocker is the static transport model's 7.5° major-width error
of `15.6738%` versus the `15%` gate. The field gate, mass gate, and sensitivity
gate pass, but the task's required W1 validation status cannot be claimed.

### Modified files

- `HANDOFF.md`
- `models/warp_air_field_v1.json`
- `scripts/validate_warp_spray.py`
- `src/aerospace_painting/warp_air_field.py`
- `src/aerospace_painting/warp_spray.py`
- `tests/test_warp_spray.py`
- `results/air_assisted/warp_w1/`
- `media/air_assisted_spray/warp_w1_air_field_fit.png`
- `media/air_assisted_spray/warp_w1_openfoam_7p5_comparison.png`
- `media/air_assisted_spray/warp_w1_angle_response.png`

### Next concrete action

Keep the Warp implementation and evidence as a failed validation checkpoint.
If refinement is authorized, audit the teacher-vs-Warp trajectory and
major-width discrepancy without fitting to the 7.5° deposition target; only
after all gates pass should Warp be considered for a separate moving-scene
integration task.

## NVIDIA Warp W1.1 teacher-forced vector-field diagnostic checkpoint (2026-09-08)

### Objective

Isolate the analytic carrier-field contribution to the W1 transport mismatch by
replaying the same CUDA particle transport kernel against the actual solved
OpenFOAM v2606 `U` fields for 0°, 7.5°, and 15°.

### Scope

- Diagnostic-only teacher-forced carrier mode; no production carrier surrogate
  and no moving Isaac runtime integration.
- Existing canonical OpenFOAM `C`/`U` outputs at time `0.06` only; no new CFD
  case, deposition fit, or physics-parameter tuning.
- Existing Warp 1.13.0/CUDA runtime and the frozen W1 high ensemble: 2,000
  particles per bin, 10,000 total, `dt=5e-5 s`, 1,200 substeps.

### Acceptance criteria

Reconstruct each solved vector field safely as a regular Cartesian grid,
confirm CPU/GPU trilinear agreement, preserve explicit out-of-field escape,
and rerun the original W1 7.5° gates with the teacher-forced field.

### Completed

- Added `warp_teacher_flow.py` with coordinate-order reconstruction, grid
  validation, finite-volume bounds, CPU trilinear sampling, and Warp GPU
  sampling.
- Added `TEACHER_FORCED_U` to the existing transport kernel while preserving
  `ANALYTIC_AIR` as the default W1 path.
- Added the W1.1 diagnostic validator, portable teacher-flow tests, compact
  NPZ fields, manifest, per-angle metrics, summary, and comparison figures.
- Used the actual installed v2606 fields. All three grids are `U[nz,ny,nx,3]`
  with dimensions `(24, 24, 48)`, first center
  `(-0.14375, -0.14375, 0.0025)`, and spacing approximately
  `(0.0125, 0.0125, 0.005)` m.

### Current checkpoint

`W1_TRANSPORT_CORE_VALIDATED_WITH_TEACHER_U`.

The teacher-forced 7.5° replay passes every original W1 gate: deposited-mass
relative error `1.75503e-6`, centroid error `0.377920 mm`, major sigma error
`5.79488%`, minor sigma error `13.876997%`, field correlation `0.963104`,
field NRMSE `0.0249804`, and relative mass-balance error `0`.

### Decisions and reasons

- Teacher-forced `U` is a diagnostic control, not a deployable model; no
  deposition or transport parameter was fitted to the 7.5° target.
- The sampler uses finite-volume half-cell bounds and marks particles outside
  the field as escaped instead of silently clamping them.
- The original W1 evidence and failed analytic result remain unchanged; the
  result isolates the primary discrepancy to the analytic carrier field.

### Verification evidence

- CUDA command: `Isaac Sim Python launcher scripts/validate_warp_w1_1_teacher_u.py
  --device cuda:0` -> exit `0`, Warp 1.13.0 on `cuda:0`, 10,000 particles per
  angle, and status `W1_TRANSPORT_CORE_VALIDATED_WITH_TEACHER_U`.
- CPU/GPU sampler agreement passed for all three fields. Maximum absolute
  velocity errors were `1.335e-5`, `7.629e-6`, and `1.335e-5` m/s with
  `inside_mask_match=true` and tolerance `2e-5` m/s.
- At 7.5°, OpenFOAM / analytic Warp / teacher-forced Warp centroid U was
  `0.0399889 / 0.0315828 / 0.0399962` m; major sigma was
  `0.0227698 / 0.0192009 / 0.0240893` m; correlation was
  `0.835503 / 0.835503 / 0.963104` and NRMSE was
  `0.0628685 / 0.0628685 / 0.0249804` for the analytic and teacher maps.
- Focused tests: `pytest -q tests/test_warp_teacher_flow.py
  tests/test_warp_spray.py tests/test_spray_frames.py
  tests/test_mass_accounting.py` -> `11 passed, 1 skipped`.
- Compilation: `python -m compileall -q src scripts` passed; `git diff --check`
  passed.
- Evidence files: `results/air_assisted/warp_w1_1/diagnostic_summary.json`,
  `teacher_flow_manifest.json`, three per-angle metrics JSON files, three
  compact teacher-field NPZ files, and
  `media/air_assisted_spray/warp_w1_1_teacher_u_7p5_comparison.png` plus
  `warp_w1_1_angle_response.png`.

### Not executed

- No production compact carrier surrogate, moving Isaac integration, Isaac
  Lab/RL, new CFD case, or deposition-model tuning was executed.
- The full legacy `pytest -q` suite was not promoted: its pre-existing
  collection errors reference missing `scripts.validate_s2_surrogate` and
  `scripts.postprocess_openfoam_flat_plate` modules outside this diagnostic.

### Blockers

None for the W1.1 diagnostic. The deployable W1 path remains blocked by the
analytic carrier field until a compact full-vector surrogate is built and
blindly rechecked at 7.5°.

### Modified files

- `HANDOFF.md`
- `src/aerospace_painting/warp_spray.py`
- `src/aerospace_painting/warp_teacher_flow.py`
- `scripts/validate_warp_w1_1_teacher_u.py`
- `tests/test_warp_teacher_flow.py`
- `results/air_assisted/warp_w1_1/`
- `media/air_assisted_spray/warp_w1_1_teacher_u_7p5_comparison.png`
- `media/air_assisted_spray/warp_w1_1_angle_response.png`

### Next concrete action

Build a compact full-vector carrier surrogate from the solved 0°/15° fields,
then rerun the original blind 7.5° gate. Keep it as a separate diagnostic
checkpoint and do not integrate it into the moving Isaac runtime yet.

## NVIDIA Warp W1.2 compact full-vector carrier checkpoint (2026-09-08)

### Objective

Replace the failed scalar analytic carrier with the specified compact
two-anchor full-vector surrogate using only the solved OpenFOAM v2606 0° and
15° fields, then evaluate the frozen blind 7.5° carrier and deposition hold-out.

### Scope

- Canonical `COROTATING_VECTOR_INTERP` over the shared `U[nz,ny,nx,3]` grid;
  `WORLD_LINEAR_DIAGNOSTIC` is retained only as an analysis mode.
- Existing Warp 1.13.0/CUDA static particle core: 10,000 particles,
  2,000/bin, five bins, `dt=5e-5 s`, 1,200 steps.
- No new CFD, no 7.5° carrier data in model construction, and no moving Isaac
  integration, plume rendering, Isaac Lab, or RL.

### Acceptance criteria

Freeze the 7.5° predicted carrier before opening the solved 7.5° `U`, validate
whole/high-speed/near-target vector errors, preserve explicit out-of-field
escape, rerun the unchanged W1 deposition gates, and compare analytic W1,
teacher-U W1.1, S2, and vector W1.2.

### Completed

- Added `warp_vector_carrier.py`, deterministic model build/provenance,
  runtime NPZ/JSON artifacts, frozen 7.5° prediction, validation script,
  portable tests, and four measured figures.
- The model contains only the 0° and 15° full-vector anchors, shared grid
  metadata, frame convention, source hashes, and source commit
  `c6a17feab4d8850bf879dfe6b8ca8cc2ad2def02`.
- Endpoint assertions pass: vector W1.2 versus teacher-U W1.1 deltas are below
  `1.0e-10 m` for centroid and below `6.0e-15 kg` for deposited mass at both
  0° and 15°.
- CPU/Warp sampler agreement passes at all three angles with maximum velocity
  error `1.90735e-6 m/s` against a `2e-5 m/s` tolerance.

### Current checkpoint

`WARP_VECTOR_CARRIER_NOT_VALIDATED` (`PARTIAL`).

The frozen blind 7.5° carrier has whole-domain normalized vector RMSE
`0.401669` versus the `0.20` band and vector-component correlation `0.877521`
versus the `0.90` band. Median direction error is `0.83136°` and passes its
band, but predicted coverage is only `84.8958%` (`4,176/27,648` cells outside
the two-anchor transformed domain).

The unchanged 7.5° deposition gates fail for vector W1.2: deposited-mass
relative error `83.2850%`, centroid error `34.6517 mm`, sigma-major error
`10.3082%`, sigma-minor error `46.9736%`, field correlation `0.244058`, field
NRMSE `0.0887899`, and relative mass balance `0`.

### Decisions and reasons

- The canonical co-rotating vector interpolation was implemented exactly as
  specified; no hold-out fitting, gate relaxation, target shrinking, or
  particle-physics tuning was applied after inspection.
- Out-of-field transformed queries are explicitly escaped. The 7.5° vector
  run escaped `8.32850e-7 kg` and deposited `1.67150e-7 kg`; this behavior is
  retained as evidence rather than hidden by clamping.
- W1.2 remains a static carrier-surrogate audit and is not promoted to the
  moving runtime.

### Verification evidence

- CUDA command: `Isaac Sim Python launcher scripts/validate_warp_w1_2.py
  --device cuda:0` -> exit `0`, Warp 1.13.0 on `cuda:0`, all 0°/7.5°/15°
  analytic, teacher-U, and vector cases executed, status
  `WARP_VECTOR_CARRIER_NOT_VALIDATED`.
- Carrier hold-out regional metrics: high-speed normalized vector RMSE
  `0.404237`, vector correlation `0.871904`, median direction error `0.60808°`;
  near-target normalized vector RMSE `0.608963`, vector correlation
  `0.761206`, median direction error `5.03959°`, p95 direction error `90°`.
- Performance at 7.5°: analytic W1 mean/p95/wall
  `0.000645068/0.001182805/0.775818 s`; teacher-U W1.1
  `0.000855671/0.001349805/1.029044 s`; vector W1.2
  `0.000610159/0.000927200/0.734327 s`, with vector model upload
  `0.000124300 s`. OpenFOAM runtime was not present in the existing static
  artifacts and was not measured.
- Focused tests: `pytest -q tests/test_warp_vector_carrier.py
  tests/test_warp_teacher_flow.py tests/test_warp_spray.py
  tests/test_spray_frames.py tests/test_mass_accounting.py` -> `15 passed,
  2 skipped`.
- Compilation and hygiene: `python -m compileall -q src scripts` passed;
  `git diff --check` passed.
- Full legacy `pytest -q` remains blocked only by its two pre-existing
  collection imports: missing `scripts.validate_s2_surrogate` and missing
  `scripts.postprocess_openfoam_flat_plate`.
- Model hashes: semantic model `eb2f29cc95a69b18877c6f01f7903c520ab8486977b2d584aedec04b140a2a22`,
  model-JSON provenance `cd91b985ca12ab7abcaf3fdaf482ced23d4017cd897676963041e3f71812ee87`,
  NPZ `f02b6d6ee664007f4c77cdd11dabb2ff6d6f7aecc0d1df0402793450dd75deb3`,
  frozen prediction `12f7e684e76c722519fc3e18235e128562cefaeb63e056bbf282a3b29799294d`.
- Media was opened and inspected: the carrier hold-out, five-way deposition
  hierarchy, metric hierarchy, and angle-response figures render correctly.

### Not executed

- No moving native Isaac integration, live spray plume, curved-surface Warp
  deposition, Isaac Lab, RL, new CFD case, or production carrier deployment.
- No attempt was made to force the 15° transfer-efficiency loss by shrinking
  the target/domain.

### Blockers

The two-anchor carrier is not validated: transformed endpoint coverage and the
blind 7.5° vector/deposition mismatch fail the engineering and original W1
gates. Do not proceed to moving-scene integration.

### Modified files

- `HANDOFF.md`
- `src/aerospace_painting/warp_spray.py`
- `src/aerospace_painting/warp_vector_carrier.py`
- `scripts/build_warp_vector_carrier.py`
- `scripts/validate_warp_w1_2.py`
- `tests/test_warp_vector_carrier.py`
- `models/warp_vector_carrier_v2.json`
- `models/warp_vector_carrier_v2.npz`
- `results/air_assisted/warp_w1_2/`
- `media/air_assisted_spray/warp_w1_2_carrier_holdout.png`
- `media/air_assisted_spray/warp_w1_2_openfoam_7p5_comparison.png`
- `media/air_assisted_spray/warp_w1_2_fidelity_hierarchy.png`
- `media/air_assisted_spray/warp_w1_2_angle_response.png`

### Next concrete action

Audit the demonstrated transformed-domain coverage and 7.5° mismatch before
designing any further carrier representation. Keep W1.2 as a failed static
checkpoint and do not integrate it into Isaac yet.

## NVIDIA Warp W1.3 fixed-grid carrier checkpoint (2026-09-09)

### Objective

Replace the failed W1.2 transformed-domain query with fixed-grid local vector
interpolation, freeze a new blind angle, and validate the unchanged transport
and deposition gates without fitting to the hold-out.

### Result

`WARP_W1_VECTOR_CARRIER_VALIDATED`.  The model uses the solved OpenFOAM v2606
0° and 15° full-vector anchors on a shared fixed grid.  The blind 10° carrier
hold-out records normalized vector RMSE `0.0258028`, vector correlation
`0.999502`, and zero out-of-field queries.  The corresponding deposition
hold-out records centroid error `1.58472 mm`, correlation `0.950426`, NRMSE
`0.0258069`, and a closed particle mass ledger.

The 7.5° result is retained as a regression comparison, not a new blind
hold-out.  The W1.2 transformed-domain failure remains preserved as historical
evidence and is not the release-candidate carrier path.

### Evidence

- JSON: `results/air_assisted/warp_w1_3/validation_summary.json`.
- Model: `models/warp_vector_carrier_v3.json` and
  `models/warp_vector_carrier_v3.npz`.
- Media: `media/air_assisted_spray/warp_w1_3_fidelity_hierarchy.png` and the
  companion carrier/deposition comparison figures.

## Native Isaac Sim W2 Warp plume checkpoint (2026-09-09)

### Objective

Integrate the validated W1.3 carrier into the existing native Isaac Sim S2
runtime as an optional GPU Lagrangian plume layer, without changing the
authoritative S2 deposition overlay or adding a new CFD case.

### Result

`ISAAC_WARP_PLUME_VALIDATED`.  The native scene completed 720 frames and
emitted 225 Warp batches with 2,500 parcels per batch.  Warp deposited
`0.00149979733333333 kg`, escaped `2.02666666666667e-7 kg`, and closed its mass
ledger with residual `5.21486e-18 kg`.  The S2 surface-map closure remained
`1.95156e-18 kg` and the S2 ledger residual remained `-5.62430e-18 kg`.

The viewer computes actual robot/TCP scene motion, interpolated W1.3 carrier
velocity, Warp droplet trajectories with drag/gravity/collision handling, and
active plume positions.  The surface overlay remains the S2 CFD-calibrated
deposited-mass density.  The moving-scene transport uses a quasi-steady local
tangent-patch approximation; it is not online CFD.

### Evidence and verification

- Native metrics: `results/air_assisted/isaac_warp_runtime/runtime_metrics.json`.
- Reviewed media: `media/air_assisted_spray/isaac_warp_plume_hero.png`,
  `isaac_warp_plume_deposition.png`, and `isaac_warp_plume_demo.mp4`.
- Portable verification remains `46 passed, 2 skipped`; compilation and
  `git diff --check` pass.  The native command was
  `Isaac Sim Python launcher scripts/run_isaac_s2_runtime.py --headless --warp-plume`.

### Release boundary

This checkpoint does not claim primary atomization, breakup, evaporation,
stochastic turbulent dispersion, splash/rebound, wall-film transport, curing,
physical film thickness, or production coating qualification.  S2 is the
authoritative fast deposition model; Warp is the optional transport/plume
layer.
