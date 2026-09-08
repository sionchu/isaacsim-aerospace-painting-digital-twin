# Spray reference solver decision

## Decision for this release candidate

The offline reference layer uses OpenCFD/Keysight OpenFOAM v2606 from the
official Debian package repository for Ubuntu 24.04 WSL2.  The Foundation
release line is not used.  The solver is kept outside the native Isaac Sim
runtime and is used to generate reproducible teacher fields and wall-deposition
maps.

## Executed reference workflow

The checked-in flat-plate cases use `sprayFoam` with five configured
`ConeNozzleInjection` bins and explicit wall stick/escape bookkeeping.  The
medium cases cover 0°, 7.5°, 10°, and 15° incidence.  Each case records its
mesh, inputs, solver completion, deposition map, and mass ledger under
`reference_cfd/openfoam_v2606/` and `results/air_assisted/openfoam_v2606/`.

The 7.5° case is the S2 hold-out.  The 10° case is the blind W1.3 carrier and
deposition hold-out.  The 0° and 15° fields are the W1.3 full-vector anchors.

## Downstream use

- S2 fits a fast deposition response from the reference maps and is validated
  on the 7.5° hold-out.
- W1.3 fits a fixed-grid full-vector carrier from the 0°/15° anchors and is
  validated on the blind 10° hold-out.
- Native W2 reuses W1.3 for optional Warp plume transport while S2 remains the
  authoritative surface deposition overlay.

OpenFOAM is not called during native Isaac Sim stepping, and no claim of
online CFD or production coating qualification is made.
