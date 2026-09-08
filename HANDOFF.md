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
- Repository verification: 25 tests passed, Python compilation passed, and
  `git diff --check` passed.

## Current level

`S2` — CFD-calibrated covariance-moment deposition surrogate validated on a
held-out 7.5-degree OpenFOAM v2606 medium case.  The calibrated interval is
`0-15 degrees`; no extrapolation or downstream runtime integration was run.

## Next checkpoint

The next concrete action is a separate task to integrate the frozen S2 API
into the native Windows runtime.  No Isaac Sim, Warp, Isaac Lab, or RL work
was executed in this checkpoint.
