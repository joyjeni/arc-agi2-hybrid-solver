"""Solver modules — auto-registers all solvers on import."""

# Importing the modules triggers @register_solver decorators
from . import (
    dsa_solvers,       # noqa: F401
    vedic_solvers,     # noqa: F401
    agentic_solvers,   # noqa: F401
)
from .registry import REGISTRY, register_solver

__all__ = ["REGISTRY", "register_solver"]
