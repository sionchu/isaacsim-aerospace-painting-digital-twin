# Optional reference CFD workflow

This directory documents the external reference layer without vendoring a CFD
solver.  The current branch stops at Level 0: no external CFD case has been
executed and no generated field or metric is checked in.

## Candidate

Iteration-CFD is the inspected candidate:

- repository: <https://github.com/using76/Iteration-CFD>
- revision: `f8027745a77406f02521e43dd61308239dd95e01`
- license: Prosperity Public License 3.0.0 plus licensor terms
- build prerequisites: Rust 1.85+, Visual Studio 2022 C++ workload, CUDA 13.x

Use it as an independently installed dependency only after the intended use
has a compatible license decision.  Do not copy its source or binaries into
this repository.

## Required first case

The first case must contain only:

```text
stationary nozzle + external air-assist source + free-air domain + flat plate
```

The case adapter must record `T_world_nozzle`, `T_world_plate`, the mesh and
timestep, droplet-bin inputs, solver commit, residual summary, wall outcomes,
and an explicit mass ledger.  It must export a surface-local deposition map in
fan-major/fan-minor `(u, v)` coordinates.

Do not use a CFD image as evidence until the mass ledger closes and a
coarse/medium sensitivity comparison has been recorded.
