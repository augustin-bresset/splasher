"""Small pure-numpy viridis colormap (no matplotlib): values -> RGBA float [0,1]."""

from __future__ import annotations

import numpy as np

# Approximate viridis anchors (0 -> 1).
_VIRIDIS = (
    np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    / 255.0
)


def colormap(values, *, vmin: float | None = None, vmax: float | None = None,
             alpha: float = 1.0, lut: np.ndarray = _VIRIDIS) -> np.ndarray:
    """Map `values` (1D) to an RGBA array `(N, 4)` float32 in [0, 1].

    `vmin`/`vmax` default to the finite min/max of the values.
    """
    v = np.asarray(values, dtype=np.float32).ravel()
    finite = np.isfinite(v)
    if vmin is None:
        vmin = float(v[finite].min()) if finite.any() else 0.0
    if vmax is None:
        vmax = float(v[finite].max()) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    t = np.where(finite, t, 0.0)

    n = len(lut) - 1
    pos = t * n
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, n)
    frac = (pos - lo)[:, None]
    rgb = lut[lo] * (1.0 - frac) + lut[hi] * frac

    out = np.empty((v.shape[0], 4), dtype=np.float32)
    out[:, :3] = rgb
    out[:, 3] = alpha
    return out
