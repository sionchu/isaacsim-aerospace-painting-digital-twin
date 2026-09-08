# Research basis for the air-assisted spray extension

This note separates physical ideas that motivate the architecture from what
the repository currently implements.

## CFD and droplet transport

**Garbero, Vanni, and Baldi (2002), “CFD modelling of a spray deposition
process of paint.”** The paper describes an Euler–Lagrange treatment for
droplet trajectories and a VOF treatment for wall impact.  It also discusses
the role of carrier air in droplet transport and overspray.

Source: <https://doi.org/10.1002/1521-3900%28200209%29187%3A1%3C719%3A%3AAID-MASY719%3E3.0.CO%3B2-U>

Applicable here: the staged separation between carrier flow, Lagrangian
droplets, wall outcomes, and a deposition ledger.  Not implemented here:
VOF wall-film spreading, atomization calibration, or a production coating
quality model.

## Surface path and orientation

**Chen and Zhao (2013), “Path Planning for Spray Painting Robot of Workpiece
Surfaces.”** The open-access article formulates a deposition-rate model and
surface path planning for free-form workpieces.

Source: <https://doi.org/10.1155/2013/659457>

Applicable here: surface-local sampling, stand-off, and the need to account
for orientation and distance.  The repository's existing geometric model is
an independent implementation and is not presented as the paper's calibrated
model.

**Wu and Tang (2023), “Robotic spray painting path planning for complex
surface: boundary fitting approach.”** The work discusses boundary-aware pass
construction, smooth path fitting, and orientation along the pass.

Source: <https://www.cambridge.org/core/journals/robotica/article/robotic-spray-painting-path-planning-for-complex-surface-boundary-fitting-approach/DF42E26DA6C84C782DCABE47D7A99B4D>

Applicable here: keeping path generation and tool orientation separate from
the later deposition model.  The current project does not claim the paper's
optimization or coating-uniformity results.

## Learning and data direction

**PaintNet (2022), “Unstructured Multi-Path Learning from 3D Point Clouds for
Robotic Spray Painting.”** This work is a reference for the future separation
between surface representation, multi-path planning, and learning.

Source: <https://arxiv.org/abs/2211.06930>

Applicable later: compact observations and path-level learning once a fast
deposition runtime has been validated.  No learning environment or policy is
claimed by the current branch.

## Scope choice

The first benchmark is intentionally a flat plate with a stationary nozzle.
The robot and generic aircraft surface remain later integration targets.  The
portable runtime is designed to support anisotropic fan footprints (major and
minor spread, rotation, and centroid shift) even if an external reference
solver initially accepts only a simpler injector representation.
