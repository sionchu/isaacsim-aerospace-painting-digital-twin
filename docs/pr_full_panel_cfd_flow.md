# feat: integrate CFD-referenced full-panel robotic painting digital twin

## Summary

This PR evolves the original aerospace robotic painting prototype into a
validated full-panel process digital twin built around OpenFOAM, NVIDIA Warp,
and Isaac Sim.

```text
OpenFOAM v2606 reference CFD
→ S2 CFD-calibrated deposition surrogate
→ NVIDIA Warp GPU Lagrangian droplet transport
→ Native Isaac Sim FANUC painting runtime
→ Full-panel deposited-mass accumulation
→ Estimated wet-film thickness (WFT)
```

The final demo executes 28 painting passes over a 2.8466 m² curved generic
aircraft surface.  The 7.5° OpenFOAM velocity field is shown in Isaac Sim as
an offline reference visualization; the CFD solve is not run live in Isaac
Sim.

## Key results

| Validation layer | Result |
|---|---|
| S2 7.5° hold-out | Centroid error 0.403 mm, correlation 0.93065, NRMSE 0.03316 |
| Warp W1.3 blind 10° | Carrier NRMSE 0.02580, carrier correlation 0.99950 |
| Warp deposition blind 10° | Centroid error 1.585 mm, correlation 0.95043, NRMSE 0.02581 |
| Full-panel process | 28 passes over 2.8466 m² |
| Estimated WFT | Mean 3.649 µm, P05–P95 2.121–3.903 µm, CV 12.6% |
| Uniformity indicator | 94.6% of area within ±20% of predicted mean |
| Native Warp runtime | 1,722 batches × 2,500 parcels = 4,305,000 computational parcel histories |
| Full-panel mass accounting | Combined residual ≈ 8.88e-16 kg |

## Fidelity boundary

The OpenFOAM CFD field is solved offline and mapped into the local process
frame in Isaac Sim.  It is not a real-time CFD solve inside Isaac Sim.

The moving Warp layer uses a quasi-steady local tangent-patch approximation.

Estimated WFT is calculated as:

```text
Estimated WFT = deposited mass / (surface area × configured liquid density)
```

Estimated WFT is model-derived, not measured or production-qualified coating
thickness.

The current model does not validate primary atomization, breakup, evaporation,
stochastic turbulent dispersion, splash/rebound, wall-film transport, sagging,
curing, physical film-thickness measurement, or production coating
qualification.

The native viewer renders actual live Warp particle positions at
`/World/AerospacePaintingCell/WarpSprayPlume` and also uses a visual-only plume
guide at `/World/AerospacePaintingCell/WarpSprayPlumeGuide`.  The guide has
`adds_mass = false`, `adds_deposition = false`, and `visual_only = true`; it
does not affect transport, deposition, forces, or mass accounting.  The visible
orange plume is therefore a composite presentation, not a claim that every
visible orange point is a physical Warp parcel.

DFT remains secondary and is labeled **Illustrative DFT estimate**, with
assumed volume solids = 50% and `synthetic_demo_only`.  It is not presented as
validated aircraft coating thickness.

## Public validation table

| Layer | Result |
|---|---|
| S2 7.5° hold-out | Centroid error 0.403 mm, corr 0.93065, NRMSE 0.03316 |
| Warp W1.3 blind 10° | Carrier NRMSE 0.02580, carrier corr 0.99950, deposition centroid 1.585 mm |
| Full-panel process | 28 passes over 2.8466 m² |
| Estimated WFT | Mean 3.649 µm, P05–P95 2.121–3.903 µm, CV 12.6% |
| Native Warp runtime | 1,722 batches × 2,500 parcels = 4,305,000 computational parcel histories |
| Mass accounting | Full-panel combined residual ~8.88e-16 kg |

## Media

- [Full-panel hero](../media/air_assisted_spray/isaac_full_panel_film_hero.png)
- [Full-panel result](../media/air_assisted_spray/isaac_full_panel_film_final.png)
- [CFD flow field still](../media/air_assisted_spray/isaac_cfd_flow_field.png)
- [CFD + Warp combined still](../media/air_assisted_spray/isaac_cfd_warp_combined.png)
- [CFD flow slice still](../media/air_assisted_spray/isaac_cfd_flow_slice.png)
- [FLOW → WARP → FULL-PANEL WFT demo](../media/air_assisted_spray/isaac_full_panel_cfd_flow_demo.mp4)

The earlier W2 media remain supporting and historical evidence.

## Synchronized spray-analysis capture

The final technical capture uses one simulated timeline for spray state,
actual Warp particle positions, mesh-hit observations, S2 finite-surface
deposition, and the Estimated WFT overlay.  The technical combined view keeps
`/World/AerospacePaintingCell/WarpSprayPlumeGuide` disabled; short particle
trails and impact markers are display-only and add zero mass.  S2 remains the
authoritative WFT layer and the projected Warp hit map is diagnostic.

- [Spray-analysis close-up](../media/air_assisted_spray/isaac_spray_analysis_closeup.png)
- [Spray-analysis combined still](../media/air_assisted_spray/isaac_spray_analysis_combined.png)
- [Progressive WFT still](../media/air_assisted_spray/isaac_progressive_wft.png)
- [Synchronized spray-analysis demo](../media/air_assisted_spray/isaac_spray_analysis_demo.mp4)

The 85 s H.264 capture is 1920×1080 at 30 fps.  The CFD direction gate
reports mean axial velocity `12.7113 m/s` and `100%` positive axial samples in
the clipped `[-0.06, 0.06] × [-0.06, 0.06] × [0, 0.24] m` nozzle-to-panel ROI.
The native run recorded 1,722 batches × 2,500 particles = 4,305,000 actual
computational parcel histories, with a maximum hit-to-overlay offset of
`0.01554 s` and zero spray-off emission violations.

## Verification

- `python -m pytest -q` → 60 passed, 2 skipped
- `python -m compileall -q src scripts tests` → passed
- `git diff --check` → passed
- Compact CFD artifact: 27,648 cells, provenance-checked U/C hashes, no NaN/Inf
- Native capture: H.264, 1920×1080, 30 fps, 85.0 s

## Release intent

The public thesis is a full-panel robotic painting digital twin combining
OpenFOAM-referenced process modeling, CFD-calibrated S2 deposition, GPU Warp
droplet transport, native Isaac Sim robot/process execution, and surface-wide
estimated wet-film thickness.  This PR changes reporting, visualization, and
release presentation; it does not add new CFD, surrogate physics, Isaac Lab,
RL, or production claims.  The synchronized spray-analysis video is the
preferred final portfolio demo.
