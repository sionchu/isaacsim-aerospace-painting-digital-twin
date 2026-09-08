# OpenFOAM v2606 reference workflow

This directory contains the external reference layer without vendoring the
OpenFOAM solver.  The cases were prepared for OpenCFD/Keysight OpenFOAM v2606
on Ubuntu 24.04 WSL2 and use the official package installation.

## Case contract

Each case contains a stationary nozzle, prescribed external air-assist source,
free-air domain, and flat plate.  The case adapter records nozzle and plate
frames, mesh and timestep, droplet-bin inputs, solver completion, wall
outcomes, and an explicit mass ledger.  It exports a surface-local deposition
map in fan-major/fan-minor `(u, v)` coordinates.

The checked-in medium cases are:

- `flat_plate/` — 0° reference;
- `flat_plate/medium_incidence_7p5deg/` — S2 hold-out;
- `flat_plate/medium_incidence_10deg/` — W1.3 blind hold-out; and
- `flat_plate/medium_incidence_15deg/` — second carrier anchor.

The corresponding metrics and maps are under
`results/air_assisted/openfoam_v2606/`.  Do not treat a CFD image as process
evidence without the matching solver completion and closed mass ledger.

## Solver boundary

OpenFOAM is an offline teacher/reference layer.  It is not called during native
Isaac Sim stepping.  S2 and W1.3 consume the checked-in reference evidence;
native W2 uses W1.3 for optional GPU plume transport while S2 remains the
authoritative surface deposition overlay.
