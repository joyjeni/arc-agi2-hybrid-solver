"""Solver modules — auto-registers all solvers on import."""

from .registry import REGISTRY, register_solver

# Importing the modules triggers @register_solver decorators
from . import dsa_solvers      # noqa: F401
from . import vedic_solvers    # noqa: F401

__all__ = ["REGISTRY", "register_solver"]
