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

The synchronized technical analysis view is stricter: the guide is disabled,
and the visible Warp particles, short trails, and impact markers are generated
from actual recorded particle positions and Warp mesh-hit events.  Those
markers and trails are display-only and add zero mass.  The live surface tint
and WFT overlay are driven continuously by the authoritative S2 finite-surface
accumulation; the projected Warp hit map is diagnostic and is not an alternate
thickness model.

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

## OpenFOAM flow visualization boundary

The canonical 7.5° OpenFOAM v2606 `U`/`C` field is solved offline and stored
as a compact, provenance-checked structured artifact for Isaac Sim.  The
native viewer renders U-derived velocity vectors, a |U| magnitude slice, and
deterministic streamlines after a quasi-steady rigid mapping into the current
local tangent process frame.  This is a reference-field visualization, not a
time-accurate full-aircraft CFD solve and not an Isaac Sim CFD solver.

During the synchronized spray-analysis capture, the same simulated timestamp
is used for spray state, Warp emission/hit observations, S2 deposition, and
the WFT overlay.  This is synchronized visualization of an offline reference
field and process model, not a live CFD solve.
