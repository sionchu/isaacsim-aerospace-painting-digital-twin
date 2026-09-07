# Fidelity boundary

This project models robot/auxiliary-axis motion, tool orientation,
stand-off distance, spray-cone geometry, and geometric surface coverage.

It does NOT simulate or validate paint atomization, droplet deposition,
airflow CFD, solvent evaporation, curing, or physical film thickness.

The displayed coverage is a process-planning visualization derived from
distance, cone membership, and incidence angle. Its scores are not coating
quality measurements, certification results, or production process parameters.

The demo also does not generate controller-specific KUKA, FANUC, ABB, or other
production robot programs. Any industrial deployment would require calibrated
robot and cell geometry, validated process parameters, safety analysis,
controller post-processing, and domain-specific qualification.
