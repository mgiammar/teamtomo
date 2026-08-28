"""Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from .atom_stack import AtomStack
from .grid import GridConfig
from .potential import (
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
)
from .utils.peng_model import get_peng_scattering_parameters

try:
    __version__ = version("torch-calculate-electrostatic-potential")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "AtomStack",
    "GridConfig",
    "__version__",
    "calculate_scattering_potential_2d",
    "calculate_scattering_potential_3d",
    "get_peng_scattering_parameters",
]
