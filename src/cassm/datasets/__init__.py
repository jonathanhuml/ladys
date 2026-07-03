"""Neural dynamics datasets."""

# from .lorenz_deprecated import Lorenz
from .lorenz_system import LorenzSystem
from .gaussian_process import GaussianProcess

__all__ = [
    "LorenzSystem",
    "GaussianProcess",
]
