# Air-assisted spray extension architecture

This extension keeps the existing geometric painting model and adds a staged
path toward a physics-informed runtime.  The stages are deliberately separated
so a reference solver is not called in every Isaac Sim or Isaac Lab step.

```text
static reference case
  -> mass ledger + wall-deposition map
  -> anisotropic footprint fit
  -> fast CPU/GPU deposition runtime
  -> moving curved-surface integration
  -> optional learning environment
```

## Current implementation

The repository now contains the portable contracts needed before a reference
run:

- `configs/air_assisted_spray.yaml` defines one deterministic demo parameter
  set and the nozzle-frame convention.
- `aerospace_painting.frames.NozzleFrame` defines a right-handed frame with
  `+Z` as spray direction, `+X` as fan-major, and `+Y` as fan-minor.
- `aerospace_painting.deposition_kernel.AnisotropicGaussian` is an analytic
  S1 footprint baseline in surface-local `(u, v)` coordinates.  It is not
  CFD-calibrated.
- `aerospace_painting.mass_ledger.MassLedger` provides explicit accounting
  fields and a closure check for future reference runs.

The existing surface sampler, normal construction, TCP path generator,
spray-cone visualization, and geometric coverage model remain unchanged.

## Static benchmark contract

The first physical case is a stationary nozzle, a stationary flat plate, a
prescribed external air-assist source, a discrete prescribed droplet
distribution, and a wall-deposition ledger.  The case must export nozzle and
plate transforms and map deposited mass into fan-local `(u, v)` coordinates.

No moving robot, aircraft surface, or factory cell belongs in this first CFD
mesh.  Those are later integration stages.

## Runtime contract after calibration

The intended future interface is a fast deposition step:

```python
deposition_step(tcp_pose, surface_state, process_params, dt) -> deposition_delta
```

The current branch does not implement this as a CFD-backed runtime because no
reference case has passed the mass-accounting and sensitivity gates.  The
analytic kernel is kept as a portable baseline for subsequent fitting and
tests.

## Fidelity labels

Use `AIR-ASSISTED SPRAY` for the process definition, `CFD REFERENCE` only for
an executed external-solver result, and `CFD-CALIBRATED MODEL` only after a
held-out comparison has been run.  Deposited mass is not paint thickness.
