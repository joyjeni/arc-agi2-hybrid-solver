"""Grid augmentation — 8 symmetry views + color permutations."""

from __future__ import annotations

from collections.abc import Iterator

from .grid_codec import Grid, copy_grid, flip_h, rotate90


def eight_views(grid: Grid) -> list[Grid]:
    """Return all 8 dihedral-group transforms of a grid."""
    g = copy_grid(grid)
    r1 = rotate90(g)
    r2 = rotate90(r1)
    r3 = rotate90(r2)
    return [g, r1, r2, r3, flip_h(g), flip_h(r1), flip_h(r2), flip_h(r3)]


def inverse_view(view_idx: int, grid: Grid) -> Grid:
    """Undo the i-th view transform."""
    # view_k = flip_h(rotate90^(k-4)(g)) for k in {4,5,6,7}
    # inverse: rotate90^(-(k-4)) ∘ flip_h  (flip_h is self-inverse)
    inv_map = {
        0: lambda g: g,
        1: lambda g: rotate90(rotate90(rotate90(g))),          # rotate270
        2: lambda g: rotate90(rotate90(g)),                    # rotate180
        3: lambda g: rotate90(g),                              # rotate90
        4: lambda g: flip_h(g),                                # flip_h self-inverse
        5: lambda g: rotate90(rotate90(rotate90(flip_h(g)))),  # rotate270 ∘ flip_h
        6: lambda g: rotate90(rotate90(flip_h(g))),            # rotate180 ∘ flip_h
        7: lambda g: rotate90(flip_h(g)),                      # rotate90  ∘ flip_h
    }
    return inv_map[view_idx](grid)


def apply_color_permutation(grid: Grid, perm: list[int]) -> Grid:
    """Remap colors using a permutation list (perm[old]=new)."""
    return [[perm[c] for c in row] for row in grid]


def yield_color_permuted_pairs(
    train_pairs: list[dict],
    n_perms: int = 4,
) -> Iterator[tuple[list[dict], list[int]]]:
    """Yield (augmented_train_pairs, permutation) for random color remappings."""
    import random
    colors = list(range(10))
    seen: set[tuple] = set()
    for _ in range(n_perms * 10):
        perm = colors[:]
        random.shuffle(perm)
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        aug = []
        for pair in train_pairs:
            aug.append({
                "input": apply_color_permutation(pair["input"], perm),
                "output": apply_color_permutation(pair["output"], perm),
            })
        yield aug, perm
        if len(seen) >= n_perms:
            break
