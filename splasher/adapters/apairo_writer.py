"""Write Splasher labels back into an apairo dataset as a per-frame channel.

Splasher labels a **BEV grid** (and, optionally, points). apairo stores per-frame
per-point arrays, so a save projects the labeling onto each reference frame's points
and writes one file per frame — a proper apairo ``preprocess`` channel that lines up,
index-for-index and timestamp-for-timestamp, with the source point cloud.

Two halves, deliberately split by their dependency:

- **preparation** (:func:`project_grid_labels`, :func:`reference_slice`) is pure numpy —
  it turns a grid raster / point labels into a per-point label array;
- **the write** (:func:`write_channel`) lazily ``import apairo`` and uses
  :class:`apairo.ChannelWriter` — the API apairo intends for labels produced *outside* it —
  which owns the on-disk format: per-frame files, ``timestamps.txt``, and registration in
  ``.apairo/channels.yaml``. So importing this module never imports apairo — only a save does.

The write is **per sequence** and **incremental**: each label frame is written under the same
stem as its reference point cloud (matched by timestamp), and ``ChannelWriter`` resumes an
existing channel, so annotating a dataset across several sessions accumulates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.grid import Grid

__all__ = ["project_grid_labels", "reference_slice", "write_channel"]


def _ts_key(t: float) -> float:
    """A stable dict key for a frame timestamp (float jitter is not our friend)."""
    return round(float(t), 6)


def project_grid_labels(points_xy: np.ndarray, raster: np.ndarray, grid: Grid,
                        ignore_id: int) -> np.ndarray:
    """Per-point labels from a BEV raster: each point takes its cell's class.

    Points outside the grid keep ``ignore_id``. ``points_xy`` is ``(N, 2)`` world XY.
    """
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ij, valid = grid.world_to_cell(xy)
    out = np.full(len(xy), ignore_id, dtype=np.int32)
    if valid.any():
        out[valid] = raster[ij[valid, 0], ij[valid, 1]]
    return out


def reference_slice(full_labels: np.ndarray, sizes: list[tuple[str, int]],
                    reference: str, ignore_id: int) -> np.ndarray | None:
    """Extract the ``reference`` channel's slice from concatenated per-point labels.

    Splasher's point labels are sized on the concatenation of a frame's cloud channels
    (fixed order). ``sizes`` is ``[(channel, n_points), …]`` in that order; we return the
    contiguous block that belongs to ``reference`` — the channel apairo aligns the written
    labels to. ``None`` if ``reference`` is not among ``sizes``.
    """
    offset = 0
    for name, n in sizes:
        if name == reference:
            block = full_labels[offset:offset + n]
            if len(block) != n:  # labels shorter than expected → treat as unlabeled
                return np.full(n, ignore_id, dtype=np.int32)
            return np.asarray(block, dtype=np.int32)
        offset += n
    return None


#: Per-frame data extensions the reference channel may use (npys → .npy, bin → .bin).
_FRAME_EXTS = (".npy", ".bin")


def _sequence_dirs(root: str | Path, reference: str) -> list[Path]:
    """Sequence directories under ``root`` that hold a ``reference`` channel.

    ``root`` may be a lone sequence (it *is* its only sequence) or a dataset root whose
    immediate subdirectories are the sequences. Filesystem-only — no apairo import needed.
    """
    root = Path(root)
    if (root / reference).is_dir():
        return [root]
    return [d for d in sorted(root.iterdir()) if d.is_dir() and (d / reference).is_dir()]


def _reference_index(root: str | Path, reference: str) -> dict[float, tuple[Path, str]]:
    """Map each reference-frame timestamp to its ``(sequence_dir, frame_stem)``.

    Read straight from the reference channel's files + ``timestamps.txt`` the way apairo's
    loader pairs them (stems sorted lexicographically, one timestamp row per stem). This is
    how a written label frame gets the *same* stem as the point cloud it labels, so the two
    channels line up file-for-file — and it is robust to dropped/windowed frames, since the
    match is by timestamp rather than position.
    """
    index: dict[float, tuple[Path, str]] = {}
    for seq_dir in _sequence_dirs(root, reference):
        cdir = seq_dir / reference
        stems = sorted(p.stem for p in cdir.iterdir()
                       if p.suffix in _FRAME_EXTS and "_" not in p.stem)
        ts_path = cdir / "timestamps.txt"
        if not stems or not ts_path.is_file():
            continue
        rows = np.atleast_1d(np.loadtxt(ts_path)).tolist()
        if len(rows) != len(stems):
            continue
        for stem, ts in zip(stems, rows, strict=True):
            index[_ts_key(ts)] = (seq_dir, stem)
    return index


def write_channel(root: str | Path, reference: str, channel: str,
                  labels_by_ts: dict[float, np.ndarray]) -> int:
    """Write ``labels_by_ts`` back into the apairo dataset at ``root`` as ``channel``.

    ``labels_by_ts`` maps a reference-frame timestamp to that frame's per-point labels. Each
    is written — via :class:`apairo.ChannelWriter`, the API apairo intends for labels
    produced *outside* it — as ``<sequence>/<channel>/<stem>.npy`` under the same stem as its
    reference frame, sharing the reference's timestamps and recording it as provenance.

    ``ChannelWriter`` resumes an existing channel, so frames written in earlier saves are
    preserved: annotating a dataset across several sessions accumulates. Returns the number
    of frames written this call.
    """
    import apairo  # lazy — the `apairo` extra must be installed to write

    index = _reference_index(root, reference)
    by_sequence: dict[Path, list[tuple[str, float, np.ndarray]]] = {}
    for ts, lab in labels_by_ts.items():
        hit = index.get(_ts_key(ts))
        if hit is None:
            continue  # no reference frame at this timestamp (windowed away / unknown)
        seq_dir, stem = hit
        by_sequence.setdefault(seq_dir, []).append((stem, float(ts), np.asarray(lab, np.int32)))

    written = 0
    for seq_dir, frames in by_sequence.items():
        with apairo.ChannelWriter(seq_dir, channel, loader="npys",
                                  timestamps_from=reference, sources=[reference]) as writer:
            for stem, ts, lab in frames:
                writer.add(lab, stem=stem, timestamp=ts)
                written += 1
    return written
