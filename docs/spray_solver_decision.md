# Spray reference solver decision

## Decision for this branch

No reference solver was selected for execution in this branch.  The highest
validated level is therefore Level 0 (architecture and portable contracts).

## Candidate A — Iteration-CFD

- Repository: <https://github.com/using76/Iteration-CFD>
- Revision inspected: `f8027745a77406f02521e43dd61308239dd95e01`
- License: Prosperity Public License 3.0.0 with additional licensor terms. It
  is source-available and is not an OSI open-source license.
- The current README describes incompressible/low-Mach flow, Lagrangian spray
  APIs, parcel drag, evaporation, and wall-impact/deposition functionality.
- The same README states that Lagrangian sprays have no case format and that
  spray support is a library API rather than a ready-to-run spray case driver.
- The repository build requires Rust 1.85+, Visual Studio 2022 C++ tooling,
  and CUDA Toolkit 13.x.  `rustc`, `cargo`, and `nvcc` were not available on
  the workstation during this run.

This is a credible future reference candidate, but using it here would require
a separately maintained case adapter and a license decision for the intended
use.  Its source and binaries are not copied into this repository.

## Candidate B — OpenFOAM

OpenFOAM was not installed or run.  A second CFD stack was not added after
Candidate A failed the executable/reference-case gate.  Introducing it would
also require a separate case adapter, a documented license choice, and an
independent static benchmark.

## Gate outcome

The static flat-plate CFD benchmark, mass ledger, coarse/medium sensitivity,
and deposition-map export were not executed.  Consequently this branch does
not publish CFD metrics, a CFD deposition map, or a CFD-calibrated surrogate.

## Re-entry condition

To continue, choose a solver whose license covers the intended use, install its
official prerequisites, build a minimal stationary flat-plate case, and record
the actual solver commit, mesh, timestep, residual log, wall outcomes, and
mass ledger before fitting the analytic kernel.
