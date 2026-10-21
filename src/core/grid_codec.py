"""Grid encoding and decoding utilities."""

from __future__ import annotations

from typing import Optional

Grid = list[list[int]]

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green",
    4: "yellow", 5: "gray", 6: "pink", 7: "orange",
    8: "azure", 9: "maroon",
}
COLOR_SYMBOL = {
    0: ".", 1: "a", 2: "b", 3: "c", 4: "d",
    5: "e", 6: "f", 7: "g", 8: "h", 9: "i",
}
SYMBOL_COLOR = {v: k for k, v in COLOR_SYMBOL.items()}


def grid_to_text(grid: Grid, mode: str = "numeric") -> str:
    """Encode a 2-D grid as a compact string for LLM prompting.

    Modes:
        numeric  — '0 1 2 | 3 4 5' (default, unambiguous)
        symbol   — '. a b | c d e' (compact, letter-coded)
        verbose  — 'black blue red | ...' (high interpretability)
    """
    rows = []
    for row in grid:
        if mode == "numeric":
            rows.append(" ".join(str(c) for c in row))
        elif mode == "symbol":
            rows.append(" ".join(COLOR_SYMBOL[c] for c in row))
        else:  # verbose
            rows.append(" ".join(COLOR_NAMES[c] for c in row))
    return " | ".join(rows)


def text_to_grid(text: str, mode: str = "numeric") -> Optional[Grid]:
    """Decode a grid string back to a 2-D int list. Returns None on failure."""
    try:
        rows = [r.strip() for r in text.strip().split("|")]
        grid: Grid = []
        for row in rows:
            if not row:
                continue
            tokens = row.split()
            if mode == "numeric":
                grid.append([int(t) for t in tokens])
            else:
                grid.append([SYMBOL_COLOR[t] for t in tokens])
        if not grid:
            return None
        # validate rectangular
        w = len(grid[0])
        if any(len(r) != w for r in grid):
            return None
        return grid
    except Exception:
        return None


def grid_dims(grid: Grid) -> tuple[int, int]:
    """Return (height, width)."""
    return len(grid), len(grid[0]) if grid else 0


def grid_equal(a: Grid, b: Grid) -> bool:
    return a == b


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def rotate90(grid: Grid) -> Grid:
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - j][i] for j in range(h)] for i in range(w)]


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def color_histogram(grid: Grid) -> dict[int, int]:
    hist: dict[int, int] = {}
    for row in grid:
        for c in row:
            hist[c] = hist.get(c, 0) + 1
    return hist


def most_common_color(grid: Grid) -> int:
    h = color_histogram(grid)
    return max(h, key=h.get)


def background_color(grid: Grid) -> int:
    return most_common_color(grid)
