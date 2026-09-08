"""Portable helpers for the aerospace painting digital-twin portfolio."""

from .coverage import compute_coverage
from .deposition_kernel import AnisotropicGaussian
from .frames import NozzleFrame
from .mass_ledger import MassLedger
from .path_generation import build_tcp_path
from .surface_sampling import sample_surface

__all__ = [
    "sample_surface",
    "build_tcp_path",
    "compute_coverage",
    "NozzleFrame",
    "AnisotropicGaussian",
    "MassLedger",
]
