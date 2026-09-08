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
- Repository verification: 14 tests passed, Python compilation passed, and
  `git diff --check` passed.

## Current level

`LEVEL 1` — stationary reference CFD benchmark complete.  The two meshes close
the injected/deposited mass ledger to better than `3e-11` relative error and
the mesh-sensitivity gate passes.

## Next checkpoint

Stop at LEVEL 1 verification.  A future task may use these verified reference
artifacts to define a higher-level calibration or moving-geometry study; no
downstream integration was performed here.
