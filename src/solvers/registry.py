"""
Solver registry — Factory pattern for registering and instantiating solvers.

Usage:
    registry = SolverRegistry()
    registry.register(MyDSASolver)
    solver = registry.get("MyDSASolver")
    all_solvers = registry.all_solvers()
"""

from __future__ import annotations

import logging
from typing import Type

from ..core.interfaces import BaseSolver

logger = logging.getLogger(__name__)


class SolverRegistry:
    """Central registry; follows Factory + Singleton patterns."""

    _instance: "SolverRegistry | None" = None

    def __new__(cls) -> "SolverRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._solvers: dict[str, Type[BaseSolver]] = {}
        return cls._instance

    def register(self, solver_cls: Type[BaseSolver]) -> None:
        self._solvers[solver_cls.__name__] = solver_cls
        logger.debug("Registered solver: %s", solver_cls.__name__)

    def get(self, name: str) -> BaseSolver:
        if name not in self._solvers:
            raise KeyError(f"Solver '{name}' not registered.")
        return self._solvers[name]()

    def all_solvers(self) -> list[BaseSolver]:
        return [cls() for cls in self._solvers.values()]

    def names(self) -> list[str]:
        return list(self._solvers.keys())


# Module-level singleton
REGISTRY = SolverRegistry()


def register_solver(cls: Type[BaseSolver]) -> Type[BaseSolver]:
    """Class decorator: auto-registers solver on import."""
    REGISTRY.register(cls)
    return cls
