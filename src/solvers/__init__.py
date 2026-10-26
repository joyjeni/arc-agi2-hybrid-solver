"""Solver modules — auto-registers all solvers on import."""

# Importing the modules triggers @register_solver decorators
from . import (
    dsa_solvers,           # noqa: F401
    vedic_solvers,         # noqa: F401
    agentic_solvers,       # noqa: F401
    generalized_solvers,   # noqa: F401
    new_solvers,           # noqa: F401
    advanced_solvers,      # noqa: F401
    advanced_solvers2,     # noqa: F401
    advanced_solvers3,     # noqa: F401
    advanced_solvers4,     # noqa: F401
    advanced_solvers5,     # noqa: F401
    advanced_solvers6,     # noqa: F401
    advanced_solvers7,     # noqa: F401
    advanced_solvers8,     # noqa: F401
    advanced_solvers9,     # noqa: F401
    advanced_solvers10,    # noqa: F401
    advanced_solvers11,    # noqa: F401
    advanced_solvers12,    # noqa: F401
    advanced_solvers13,    # noqa: F401
    advanced_solvers14,    # noqa: F401
    advanced_solvers15,    # noqa: F401
    advanced_solvers16,    # noqa: F401
    advanced_solvers17,    # noqa: F401
    advanced_solvers18,    # noqa: F401
    advanced_solvers19,    # noqa: F401
    advanced_solvers20,    # noqa: F401
    advanced_solvers21,    # noqa: F401
    advanced_solvers22,    # noqa: F401
    advanced_solvers23,    # noqa: F401
)
from .registry import REGISTRY, register_solver

__all__ = ["REGISTRY", "register_solver"]
