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

The native viewer renders actual live Warp particle positions at
`/World/AerospacePaintingCell/WarpSprayPlume` and also uses the visual-only
`/World/AerospacePaintingCell/WarpSprayPlumeGuide` to keep sub-pixel transport
readable in the final 1080p capture.  The guide has `adds_mass = false`,
`adds_deposition = false`, and `visual_only = true`; it does not affect
transport, deposition, forces, or mass accounting.  Not every visible orange
plume point is therefore a physical Warp parcel.

## What these layers model

- prescribed air-assisted carrier flow from the reference data;
- a prescribed discrete droplet-size distribution;
- droplet drag and gravity;
- mesh collision and deposited/escaped-mass bookkeeping;
- stand-off, incidence angle, and surface-local deposition coordinates;
- robot and auxiliary-axis motion in the native Isaac Sim scene; and
- a CFD-calibrated fast deposition footprint after held-out validation.

## Full-panel WFT and DFT wording

Estimated WFT is computed as `deposited mass / (surface area × configured
liquid density)`.  Estimated WFT is model-derived, not measured.  DFT is only an
**Illustrative DFT estimate**, with assumed volume solids = 50% and the label
`synthetic_demo_only`; it is not validated aircraft coating thickness.

## Not automatically modeled or qualified

- primary atomization;
- breakup;
- calibration of real paint rheology or manufacturer gun internals;
- stochastic turbulent dispersion;
- splash/rebound;
- full two-way coupling;
- evaporation;
- wall-film transport;
- sagging;
- curing;
- physical film thickness measurement;
- production coating quality; and
- production qualification.

Deposited mass and normalized deposited-mass density are process-model
quantities, not paint thickness.  The moving Warp integration remains
quasi-steady local tangent-patch transport; it is not online CFD.
