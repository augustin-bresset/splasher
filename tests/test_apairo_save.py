"""apairo write-back: label preparation, Session.save_apairo, and (if apairo is
installed) a real round-trip. The unit tests need no apairo — the writer is only
exercised for real in the guarded integration test at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from splasher.adapters.apairo_source import ApairoSource
from splasher.adapters.apairo_writer import project_grid_labels, reference_slice
from splasher.core.grid import Grid
from splasher.engine import Session


# --------------------------------------------------------------- pure prep
def test_project_grid_labels_takes_cell_class():
    grid = Grid(0.0, 4.0, 0.0, 4.0, 1.0)          # 4x4 cells
    raster = grid.empty_raster(0)
    raster[1, 2] = 7                               # cell (row=1, col=2) → world x∈[2,3), y∈[1,2)
    pts = np.array([[2.5, 1.5],                    # inside cell (1,2) → 7
                    [0.5, 0.5],                    # cell (0,0) → 0 (unlabeled)
                    [99.0, 99.0]])                 # outside the grid → ignore
    out = project_grid_labels(pts, raster, grid, ignore_id=0)
    assert out.tolist() == [7, 0, 0]


def test_reference_slice_extracts_the_right_block():
    full = np.array([1, 1, 1, 2, 2, 9], dtype=np.int32)   # lidarA(3) + lidarB(2) + extra(1)
    sizes = [("lidarA", 3), ("lidarB", 2), ("extra", 1)]
    assert reference_slice(full, sizes, "lidarB", 0).tolist() == [2, 2]
    assert reference_slice(full, sizes, "lidarA", 0).tolist() == [1, 1, 1]
    assert reference_slice(full, sizes, "absent", 0) is None


# --------------------------------------------------------- fake apairo source
@dataclass
class _FakeSample:
    data: dict
    timestamp: float | None = None


class _FakeDataset:
    """Two lidar frames (points in [0,4)²), synchronous, each with a timestamp."""

    is_synchronous = True

    def __init__(self):
        self.keys = ["lidar"]
        rng = np.random.RandomState(0)
        self._frames = [
            {"lidar": (rng.rand(20, 4) * np.array([4, 4, 1, 1])).astype(np.float32)}
            for _ in range(2)
        ]
        self._ts = [1000.0, 1000.1]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return _FakeSample(self._frames[i], timestamp=self._ts[i])


def _fake_apairo_session():
    src = ApairoSource(_FakeDataset(), dataset_root="/nonexistent/dset", reference="lidar")
    return Session(src), src


def test_apairo_meta_surfaces_reference_and_write_root():
    _, src = _fake_apairo_session()
    meta = src.apairo_meta()
    assert meta["is_apairo"] is True
    assert meta["reference"] == "lidar"
    assert meta["point_channels"] == ["lidar"]
    assert meta["write_root"] == "/nonexistent/dset"


def test_save_apairo_grid_projects_painted_cells(monkeypatch):
    import splasher.adapters.apairo_writer as W

    captured = {}

    def _cap(root, ref, ch, by_ts):
        captured.update(root=root, ref=ref, ch=ch, by_ts=by_ts)
        return len(by_ts)

    monkeypatch.setattr(W, "write_channel", _cap)

    session, _ = _fake_apairo_session()
    cls = session.labelset.paintable[0].id
    session.set_active_class(cls)
    session.set_frame(0)
    session.paint_rect((-1e3, -1e3, 1e3, 1e3))     # paint the whole grid on frame 0

    out = session.save_apairo(channel="ground_truth", reference="lidar", mode="grid")
    assert out["channel"] == "ground_truth" and out["mode"] == "grid"
    assert captured["ref"] == "lidar" and captured["root"] == "/nonexistent/dset"
    assert set(captured["by_ts"]) == {1000.0}       # only the painted frame is written
    lab = captured["by_ts"][1000.0]
    assert lab.shape == (20,) and np.all(lab == cls)  # every in-grid point took the class


def test_save_apairo_points_writes_reference_slice(monkeypatch):
    import splasher.adapters.apairo_writer as W

    captured = {}
    monkeypatch.setattr(W, "write_channel",
                        lambda root, ref, ch, by_ts: (captured.update(by_ts=by_ts) or len(by_ts)))

    session, _ = _fake_apairo_session()
    cls = session.labelset.paintable[0].id
    session.set_active_class(cls)
    session.set_active_targets({"points"})
    session.set_frame(1)
    session.paint_rect((-1e3, -1e3, 1e3, 1e3))     # paint every point of frame 1

    session.save_apairo(reference="lidar", mode="points")
    assert set(captured["by_ts"]) == {1000.1}
    assert np.all(captured["by_ts"][1000.1] == cls)


def test_save_apairo_rejects_non_apairo_source():
    from splasher.demo import make_demo_source

    session = Session(make_demo_source(n_frames=2))
    assert session.apairo_meta() == {"is_apairo": False}
    with pytest.raises(ValueError, match="not an apairo dataset"):
        session.save_apairo()


# ------------------------------------------------- real apairo round-trip (guarded)
def test_apairo_real_roundtrip(tmp_path):
    apairo = pytest.importorskip("apairo", reason="`apairo` extra not installed")

    # Build a tiny single-sequence raw dataset: lidar, 3 frames.
    lid = tmp_path / "lidar"
    lid.mkdir()
    ts = []
    for i in range(3):
        pts = np.zeros((10, 4), np.float32)
        pts[:, 0] = np.linspace(0.1, 3.9, 10)      # x spread across the grid
        pts[:, 1] = 1.5
        np.save(lid / f"{i:06d}.npy", pts)
        ts.append(2000.0 + i * 0.1)
    np.savetxt(lid / "timestamps.txt", ts)

    src = ApairoSource.from_path(str(tmp_path), reference="lidar", tolerance=0.05)
    session = Session(src)
    cls = session.labelset.paintable[0].id
    session.set_active_class(cls)
    session.set_frame(1)
    session.paint_rect((-1e3, -1e3, 1e3, 1e3))     # label frame 1 only

    report = session.save_apairo(channel="ground_truth", reference="lidar", mode="grid")
    assert report["frames"] == 1

    # Only the labeled frame is written, under the *same stem* as its lidar frame (000001),
    # and the channel is registered as an apairo preprocess channel.
    gt = tmp_path / "ground_truth"
    assert sorted(p.name for p in gt.glob("*.npy")) == ["000001.npy"]
    assert np.unique(np.load(gt / "000001.npy")).tolist() == [cls]
    import yaml

    reg = yaml.safe_load((tmp_path / ".apairo" / "channels.yaml").read_text())["channels"]
    assert reg["ground_truth"]["loader"] == "npys"
    assert reg["ground_truth"]["timestamps_from"] == "lidar"

    # Read back through apairo: synchronizing on the sparse channel keeps the labeled frame.
    ds = apairo.RawDataset(str(tmp_path), keys=["lidar", "ground_truth"])
    sv = ds.synchronize(reference="ground_truth", tolerance=0.05)
    assert len(sv) == 1
    assert round(sv[0].timestamp, 4) == 2000.1
    assert np.unique(sv[0].data["ground_truth"]).tolist() == [cls]


# --------------------------------------------------------------- server routes
def test_apairo_routes(monkeypatch):
    pytest.importorskip("fastapi", reason="`api` extra not installed")
    pytest.importorskip("httpx", reason="httpx required by fastapi.testclient")
    from fastapi.testclient import TestClient

    import splasher.adapters.apairo_writer as W
    from splasher.server import create_app

    captured = {}

    def _capture(root, ref, ch, by_ts):
        captured.update(ch=ch, by_ts=by_ts)
        return len(by_ts)

    monkeypatch.setattr(W, "write_channel", _capture)

    session, _ = _fake_apairo_session()
    client = TestClient(create_app(session))

    info = client.get("/api/apairo/info").json()
    assert info["is_apairo"] is True and info["reference"] == "lidar"

    cls = session.labelset.paintable[0].id
    client.post("/api/class", json={"id": cls})
    client.post("/api/paint", json={"rect": [-1e3, -1e3, 1e3, 1e3]})
    r = client.post("/api/apairo/save", json={"channel": "gt", "reference": "lidar", "mode": "grid"})
    assert r.status_code == 200 and r.json()["channel"] == "gt"
    assert captured["ch"] == "gt"

    # a bad reference is a client error, not a 500
    assert client.post("/api/apairo/save", json={"reference": "nope"}).status_code == 422


def test_apairo_info_false_for_plain_source():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from splasher.demo import make_demo_source
    from splasher.server import create_app

    client = TestClient(create_app(Session(make_demo_source(n_frames=2))))
    assert client.get("/api/apairo/info").json() == {"is_apairo": False}
