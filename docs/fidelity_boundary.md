# Geometric baseline boundary

This document describes the original portable geometric layer: robot and
auxiliary-axis motion, tool orientation, stand-off distance, spray-cone
membership, and geometric surface coverage.

The release candidate also contains a separate CFD-backed spray extension.
Its validated process layers and explicit limitations are documented in the
[spray fidelity boundary](spray_fidelity_boundary.md).  The geometric coverage
score and the CFD-derived deposited-mass density are different quantities.

Neither layer claims production coating quality, paint atomization from first
principles, physical film thickness, curing, or controller-specific production
robot programs.  Industrial deployment would require calibrated robot and cell
geometry, validated process parameters, safety analysis, controller
post-processing, and domain-specific qualification.
