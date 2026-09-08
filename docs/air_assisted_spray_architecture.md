# Air-assisted spray architecture

The release candidate separates reference CFD, fast deposition, GPU transport,
and native scene integration so the reference solver is not called in every
Isaac Sim step.

```text
OpenFOAM v2606 reference/teacher cases
  -> carrier fields + wall-deposition maps + mass ledger
  -> S2 CFD-calibrated deposition surrogate
  -> W1.3 fixed-grid full-vector carrier + Warp Lagrangian transport
  -> native Isaac Sim robot/scene/process integration
  -> optional plume visualization and future data workflows
```

## Layer responsibilities

### OpenFOAM v2606

The offline reference layer contains stationary flat-plate cases at 0°, 7.5°,
10°, and 15°.  It supplies the carrier fields, wall-deposition maps, and
mass-ledger evidence used by the downstream models.  The checked-in cases and
metrics are under `reference_cfd/openfoam_v2606/` and
`results/air_assisted/openfoam_v2606/`.

### S2 deposition surrogate

S2 fits the validated deposition moments and transfer response with the
existing surface-local kernel.  Its 7.5° hold-out is the fast runtime
validation target.  S2 is the authoritative surface deposition and mass-density
overlay in the native runtime.

### Warp transport

W1.3 interpolates a fixed-grid, full-vector carrier model built from the 0° and
15° OpenFOAM anchors and passes a blind 10° carrier/deposition hold-out.  W2
uses that model for deterministic GPU Lagrangian droplet transport with drag,
gravity, mesh collision, and explicit mass accounting.  Warp is an optional
higher-fidelity transport/plume layer; it does not replace S2 deposition.

### Native Isaac Sim

The native runtime resolves the actual composed painting-cell prims and reads
the live robot/TCP and spray-tool transforms.  It drives the moving scene,
renders the active plume, and keeps the S2 deposition overlay on the workpiece.
The moving-scene Warp transport uses a quasi-steady local tangent-patch
approximation rather than online CFD.

## Runtime contract

The portable deposition interface remains:

```python
deposition_step(tcp_pose, surface_state, process_params, dt) -> deposition_delta
```

The native adapter supplies live transforms and visualization state while the
numeric S2 and Warp modules remain testable without Isaac Sim.

## Fidelity labels

Use `AIR-ASSISTED SPRAY` for the process definition, `CFD REFERENCE` for the
executed OpenFOAM v2606 cases, `CFD-CALIBRATED MODEL` for S2 and the validated
W1.3 comparison, and `GPU LAGRANGIAN TRANSPORT` for the Warp layer.  Deposited
mass is not paint thickness, and the project does not claim production coating
qualification.
