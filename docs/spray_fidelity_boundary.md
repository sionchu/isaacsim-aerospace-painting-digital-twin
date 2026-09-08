# Spray fidelity boundary

## Validated process layers

- OpenFOAM/Keysight v2606 is the offline reference/teacher for the stationary
  flat-plate carrier and wall-deposition cases.
- S2 is a CFD-calibrated fast deposition surrogate, validated against the
  held-out 7.5° OpenFOAM case.
- NVIDIA Warp W1.3 is a fixed-grid full-vector carrier surrogate with
  Lagrangian droplet transport, validated on a blind 10° hold-out.
- Native Isaac Sim W2 uses those layers in the moving scene.  Its Warp plume
  uses a quasi-steady local tangent-patch approximation, while the S2 surface
  deposition overlay remains authoritative.

## What these layers model

- prescribed air-assisted carrier flow from the reference data;
- a prescribed discrete droplet-size distribution;
- droplet drag and gravity;
- mesh collision and deposited/escaped-mass bookkeeping;
- stand-off, incidence angle, and surface-local deposition coordinates;
- robot and auxiliary-axis motion in the native Isaac Sim scene; and
- a CFD-calibrated fast deposition footprint after held-out validation.

## Not automatically modeled or qualified

- primary atomization or breakup from first principles;
- calibration of real paint rheology or manufacturer gun internals;
- stochastic turbulent dispersion;
- droplet coalescence, splash, or rebound unless separately supported and
  checked;
- full two-way coupling;
- evaporation, wet-film leveling, sagging, dripping, or re-entrainment;
- wall-film transport, solvent chemistry, or curing;
- orange peel, dry-film thickness, or physical film-thickness measurement; and
- production coating quality or paint-quality certification.

Deposited mass and normalized deposited-mass density are process-model
quantities, not paint thickness.  The moving-scene Warp layer is a transport
and plume visualization layer; it is not online CFD.
