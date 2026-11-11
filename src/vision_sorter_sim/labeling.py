"""Connected-component labelling (CCL) from scratch.

Two-pass algorithm with a union-find equivalence table (Rosenfeld &
Pfaltz 1966): the first raster pass assigns provisional labels and
records equivalences between neighbouring labels; the second pass
replaces each provisional label by its set representative. Supports
4- and 8-connectivity; 8 is the default because diagonal contact is
still the same physical part on a conveyor.
"""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Union-find with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# Neighbour offsets (dy, dx) already visited in a raster scan.
_NEIGHBOURS_4 = ((-1, 0), (0, -1))
_NEIGHBOURS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1))


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label the foreground (True) pixels of a boolean mask.

    Returns ``(labels, n_labels)`` where ``labels`` is an int32 array
    with 0 = background and 1..n_labels the components.
    """
    if mask.dtype != bool:
        raise ValueError("label_components expects a boolean mask")
    if connectivity == 4:
        neighbours = _NEIGHBOURS_4
    elif connectivity == 8:
        neighbours = _NEIGHBOURS_8
    else:
        raise ValueError("connectivity must be 4 or 8")

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    uf = UnionFind()
    next_label = 1

    # First pass: provisional labels + equivalence recording. Only
    # foreground rows are visited, via flatnonzero, to stay fast.
    for y in range(height):
        xs = np.flatnonzero(mask[y])
        for x in xs:
            x = int(x)
            neighbour_labels = []
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    lab = int(labels[ny, nx])
                    if lab:
                        neighbour_labels.append(lab)
            if not neighbour_labels:
                labels[y, x] = next_label
                uf.make_set(next_label)
                next_label += 1
            else:
                smallest = min(neighbour_labels)
                labels[y, x] = smallest
                for other in neighbour_labels:
                    if other != smallest:
                        uf.union(smallest, other)

    if next_label == 1:
        return labels, 0

    # Second pass: resolve equivalences to compact 1..n labels.
    roots = {}
    remap = np.zeros(next_label, dtype=np.int32)
    for lab in range(1, next_label):
        root = uf.find(lab)
        if root not in roots:
            roots[root] = len(roots) + 1
        remap[lab] = roots[root]
    labels = remap[labels]
    return labels, len(roots)


def component_pixels(labels: np.ndarray, n_labels: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(ys, xs)`` coordinate arrays for each component label."""
    out = []
    for lab in range(1, n_labels + 1):
        ys, xs = np.nonzero(labels == lab)
        out.append((ys, xs))
    return out
