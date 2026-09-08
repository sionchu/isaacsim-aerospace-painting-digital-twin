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
- Repository verification: 18 tests passed, Python compilation passed, and
  `git diff --check` passed.

## Current level

`LEVEL 1` — stationary reference CFD benchmark and 15-degree incidence
response validation complete.  Nominal mesh centroid noise is
`2.8856031658e-4 m`; the incidence centroid shift is `8.0192400075e-2 m`
(`277.9x` the nominal noise), with a positive `u` shift and a passing
distinguishability gate.

## Next checkpoint

Stop at LEVEL 1 verification.  Do not proceed to calibration or downstream
integration without a separate task.
