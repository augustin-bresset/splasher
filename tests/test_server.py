"""API integration tests (skipped if the `api` extra is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from splasher.demo import make_demo_source
from splasher.engine import Session
from splasher.server.protocol import decode_array

pytest.importorskip("fastapi", reason="`api` extra not installed")
pytest.importorskip("httpx", reason="httpx required by fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from splasher.server import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    session = Session(make_demo_source(n_frames=4, seed=2))
    return TestClient(create_app(session))


def test_get_session(client: TestClient) -> None:
    r = client.get("/api/session")
    assert r.status_code == 200
    assert r.json()["n_frames"] == 4


def test_web_front_served_without_shadowing_api(client: TestClient) -> None:
    # The front is mounted on "/", but /api/* (and /docs) keep priority.
    root = client.get("/")
    assert root.status_code == 200 and "Splasher" in root.text
    assert client.get("/api/view").status_code == 200
    assert client.get("/docs").status_code == 200


def test_get_view_has_points(client: TestClient) -> None:
    d = client.get("/api/view").json()
    points = decode_array(d["points"])
    assert points.shape[1] >= 3


def test_paint_via_api_sets_grid_labels(client: TestClient) -> None:
    d = client.post("/api/paint", json={"rect": [-5, -5, 5, 5]}).json()
    grid_labels = decode_array(d["grid_labels"])
    assert grid_labels is not None and (grid_labels != 0).any()


def test_frame_and_accum_roundtrip(client: TestClient) -> None:
    client.post("/api/frame", json={"index": 2})
    n0 = len(decode_array(client.get("/api/view").json()["points"]))
    d = client.post("/api/accum", json={"radius": 2}).json()
    assert d["index"] == 2 and d["accum_radius"] == 2
    assert len(decode_array(d["points"])) > n0


def test_select_apply_clears_selection(client: TestClient) -> None:
    client.post("/api/tool", json={"tool": "select"})
    d = client.post("/api/select", json={"rect": [-3, -3, 3, 3]}).json()
    assert decode_array(d["selection"]) is not None
    d = client.post("/api/selection/apply").json()
    assert d["selection"] is None


def test_file_viewer_empty_session_and_fs(tmp_path) -> None:
    from splasher import ArraySource

    c = TestClient(create_app(ArraySource([], [])))   # empty = file-viewer mode
    assert c.get("/api/session").json()["n_frames"] == 0
    assert decode_array(c.get("/api/view").json()["points"]).shape == (0, 3)  # no clouds, no features

    np.save(tmp_path / "scan.npy", np.random.rand(20, 4).astype(np.float32))
    (tmp_path / "note.txt").write_text("nope")

    listing = c.get("/api/fs", params={"path": str(tmp_path)}).json()
    names = {e["name"]: e["openable"] for e in listing["entries"]}
    assert names["scan.npy"] is True and names["note.txt"] is False

    opened = c.post("/api/fs/open", json={"path": str(tmp_path / "scan.npy")}).json()
    assert opened["kind"] == "cloud" and decode_array(opened["points"]).shape == (20, 4)

    # a .npy of shape (H, W, 4) is an image, not a cloud
    np.save(tmp_path / "pic.npy", (np.random.rand(8, 10, 4) * 255).astype(np.uint8))
    img = c.post("/api/fs/open", json={"path": str(tmp_path / "pic.npy")}).json()
    assert img["kind"] == "image" and decode_array(img["image"]).shape == (8, 10, 4)

    bad = c.post("/api/fs/open", json={"path": str(tmp_path / "note.txt")})
    assert bad.status_code == 422 and "unsupported" in bad.json()["detail"]

    # loaded cloud becomes the labelable session source; grid labels then export to a file
    src = c.post("/api/source/files", json={"paths": [str(tmp_path / "scan.npy")]}).json()
    assert decode_array(src["points"]).shape[0] == 20
    c.post("/api/paint", json={"rect": [-100, -100, 100, 100]})
    r = c.post("/api/export", json={"dir": str(tmp_path), "name": "scan_bev.npy"})
    assert r.status_code == 200 and r.json()["path"].endswith("scan_bev.npy")
    assert (tmp_path / "scan_bev.npy").exists()


def test_save_load_via_api(client: TestClient, tmp_path) -> None:
    client.post("/api/paint", json={"rect": [-5, -5, 5, 5]})
    r = client.post("/api/save", json={"dir": str(tmp_path)})
    assert r.status_code == 200 and r.json()["ok"]
    d = client.post("/api/load", json={"dir": str(tmp_path)}).json()
    assert decode_array(d["grid_labels"]) is not None


def test_open_file_gathers_sibling_features(tmp_path) -> None:
    """File viewer (no apairo): a cloud + its `<base>_<suffix>.npy` siblings load together as
    `[x, y, z, *feature_names]`, opening either the coordinate file or a feature file."""
    from splasher.server.files import open_file

    xyz = np.random.rand(6, 3).astype(np.float32)
    inten = np.arange(6, dtype=np.uint8)
    rng = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    np.save(tmp_path / "000000.npy", xyz)
    np.save(tmp_path / "000000_intensity.npy", inten)
    np.save(tmp_path / "000000_range.npy", rng)

    res = open_file(str(tmp_path / "000000.npy"))          # open the coordinate cloud
    assert res["kind"] == "cloud" and res["feature_names"] == ["intensity", "range"]
    assert res["points"].shape == (6, 5)
    np.testing.assert_allclose(res["points"][:, 3], inten)
    np.testing.assert_allclose(res["points"][:, 4], rng)

    res2 = open_file(str(tmp_path / "000000_intensity.npy"))   # open a lone (N,) feature file
    assert res2["feature_names"] == ["intensity", "range"] and res2["points"].shape == (6, 5)
    assert res2["name"] == "000000_intensity.npy"         # the panel keeps the opened file's name


def test_open_file_lone_scalar_without_coords_errors(tmp_path) -> None:
    from splasher.server.files import open_file

    np.save(tmp_path / "lonely_intensity.npy", np.arange(4, dtype=np.uint8))
    with pytest.raises(ValueError, match="no coordinate cloud"):
        open_file(str(tmp_path / "lonely_intensity.npy"))
