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

## Current level

`LEVEL 0` — architecture and portable contracts.  No external CFD case has
been executed, so no CFD metrics, deposition map, or calibrated surrogate is
published.

## Next checkpoint

Choose and license an external reference solver for the intended use, install
its official prerequisites, and execute the stationary flat-plate benchmark.
Do not add learning or moving-aircraft integration before the mass ledger and
coarse/medium sensitivity checks pass.
