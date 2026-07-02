"""FastAPI app: drives an `engine.Session` over REST and serves the web front.

Each command returns the updated `ViewState` so a front renders in one round-trip.
Operations are synchronous (numpy): a single shared `Session` is served. The
"grid already labeled" confirmation is left to the front (see `/api/grid/labelled_count`).
"""

from __future__ import annotations

from pathlib import Path

from ..core.array_source import ArraySource
from ..core.source import ChannelKind, ChannelSpec
from ..engine.session import Session
from .files import combine_clouds, list_dir, open_file
from .protocol import encode_array, grid_from_dict, session_info_to_dict, view_state_to_dict

# Web front (vanilla, zero build) served as-is — packaged at `splasher/web/`.
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _as_session(session_or_source, labels) -> Session:
    if isinstance(session_or_source, Session):
        return session_or_source
    return Session(session_or_source, labelset=labels)


def create_app(session_or_source, *, labels=None):
    from pathlib import Path as _Path

    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    session = _as_session(session_or_source, labels)
    app = FastAPI(title="Splasher API", version="1.0")

    # The front may be served from another origin (dev): allow everything.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    def view() -> dict:
        return view_state_to_dict(session.view_state())

    # ----------------------------------------------------------- reads
    @app.get("/api/session")
    def get_session() -> dict:
        return session_info_to_dict(session.info())

    @app.get("/api/view")
    def get_view() -> dict:
        return view()

    @app.get("/api/grid/labelled_count")
    def grid_labelled_count() -> dict:
        return {"count": session.grid_labelled_count()}

    # ----------------------------------------------------------- settings
    @app.post("/api/frame")
    def set_frame(payload: dict = Body(...)) -> dict:
        session.set_frame(int(payload["index"]))
        return view()

    @app.post("/api/class")
    def set_class(payload: dict = Body(...)) -> dict:
        session.set_active_class(int(payload["id"]))
        return view()

    @app.post("/api/tool")
    def set_tool(payload: dict = Body(...)) -> dict:
        session.set_tool(str(payload["tool"]))
        return view()

    @app.post("/api/targets")
    def set_targets(payload: dict = Body(...)) -> dict:
        session.set_active_targets(payload["targets"])
        return view()

    @app.post("/api/accum")
    def set_accum(payload: dict = Body(...)) -> dict:
        session.set_accum_radius(int(payload["radius"]))
        return view()

    @app.post("/api/bev_mode")
    def set_bev_mode(payload: dict = Body(...)) -> dict:
        session.set_bev_mode(str(payload["mode"]))
        return view()

    @app.post("/api/labelset")
    def set_labelset(payload: dict = Body(...)) -> dict:
        try:
            session.set_labelset(payload)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"invalid labelset: {e}") from e
        return view()

    @app.post("/api/visibility")
    def set_visibility(payload: dict = Body(...)) -> dict:
        if "clouds" in payload:
            session.set_visible_clouds(payload["clouds"])
        if "images" in payload:
            session.set_visible_images(payload["images"])
        return view()

    # ----------------------------------------------------------- labeling
    def _rect(payload: dict):
        r = payload["rect"]
        if len(r) != 4:
            raise HTTPException(422, "rect expects [x0, y0, x1, y1]")
        return tuple(float(v) for v in r)

    @app.post("/api/paint")
    def paint(payload: dict = Body(...)) -> dict:
        session.paint_rect(_rect(payload))
        return view()

    @app.post("/api/erase")
    def erase(payload: dict = Body(...)) -> dict:
        session.erase_rect(_rect(payload))
        return view()

    @app.post("/api/select")
    def select(payload: dict = Body(...)) -> dict:
        session.select_rect(_rect(payload), op=str(payload.get("op", "add")))
        return view()

    @app.post("/api/selection/apply")
    def apply_selection() -> dict:
        session.apply_selection()
        return view()

    @app.post("/api/selection/clear")
    def clear_selection() -> dict:
        session.clear_selection()
        return view()

    @app.post("/api/clear")
    def clear_frame() -> dict:
        session.clear_frame()
        return view()

    @app.post("/api/undo")
    def undo() -> dict:
        session.undo()
        return view()

    # ----------------------------------------------------------- grid / io
    @app.post("/api/grid")
    def commit_grid(payload: dict = Body(...)) -> dict:
        try:
            grid = grid_from_dict(payload)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, f"invalid grid: {e}") from e
        session.commit_grid(grid)
        return view()

    @app.post("/api/save")
    def save(payload: dict = Body(...)) -> dict:
        out = session.save(payload["dir"])
        return {"ok": True, "dir": str(out)}

    @app.post("/api/export")
    def export_bev(payload: dict = Body(...)) -> dict:
        """Export the current BEV grid raster as a single .npy file."""
        import numpy as np

        out = _Path(payload["dir"]).expanduser() / payload["name"]
        if out.suffix != ".npy":
            out = out.with_suffix(".npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, session.grid_target.raster(session.index))
        return {"ok": True, "path": str(out)}

    @app.post("/api/load")
    def load(payload: dict = Body(...)) -> dict:
        try:
            session.load(payload["dir"])
        except (FileNotFoundError, OSError) as e:
            raise HTTPException(404, f"cannot load: {e}") from e
        return view()

    # ----------------------------------------------------------- file viewer (browse + open)
    @app.get("/api/fs")
    def fs_list(path: str | None = None) -> dict:
        try:
            return list_dir(path)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/fs/open")
    def fs_open(payload: dict = Body(...)) -> dict:
        try:
            res = open_file(payload["path"], features=payload.get("features"))
        except (FileNotFoundError, ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        if res["kind"] == "cloud":
            return {"kind": "cloud", "name": res["name"], "path": res["path"],
                    "points": encode_array(res["points"]),
                    "feature_names": res.get("feature_names", [])}
        if res["kind"] == "feature":
            # a lone per-point measure — the front may attach it to an open cloud of
            # matching length (re-opening the cloud with `features=[this path]`)
            return {"kind": "feature", "name": res["name"], "path": res["path"],
                    "length": res["length"]}
        out = {"kind": "image", "name": res["name"], "path": res["path"]}
        if "image" in res:                 # a numpy image array (e.g. .npy HxWxC)
            out["image"] = encode_array(res["image"])
        return out

    @app.get("/api/fs/raw")
    def fs_raw(path: str):
        p = _Path(path).expanduser()
        if not p.is_file():
            raise HTTPException(404, "not a file")
        return FileResponse(str(p))

    @app.post("/api/source/files")
    def set_source_files(payload: dict = Body(...)) -> dict:
        """File-viewer: make the selected cloud files the (labelable) session source.

        `paths` entries are either a plain path or `{path, features: [paths]}` (explicitly
        attached per-point measures). Combines everything into one frame so the BEV grid +
        labeling work on the loaded cloud(s); each feature becomes a `lidar_<name>` scalar
        channel, so the BEV underlay can color by it.
        """
        clouds: list[tuple] = []
        for entry in payload.get("paths", []):
            path = entry if isinstance(entry, str) else entry.get("path")
            feats = None if isinstance(entry, str) else entry.get("features")
            if not path:
                continue
            try:
                res = open_file(path, features=feats)
            except (FileNotFoundError, ValueError, OSError):
                if not feats:
                    continue
                try:
                    res = open_file(path)   # a measure file went bad → keep the bare cloud
                except (FileNotFoundError, ValueError, OSError):
                    continue
            if res.get("kind") == "cloud":
                clouds.append((res["points"], res["feature_names"]))
        keep = session.grid_labelled_count() > 0   # once you've labeled, the grid is locked
        if not clouds:
            session.set_source(ArraySource([], []), keep_grid=keep)
            return view()
        pts, names = combine_clouds(clouds)
        channels = {"lidar": pts[:, :3]}
        specs = [ChannelSpec("lidar", ChannelKind.POINTCLOUD, pts.dtype, (None, 3))]
        for i, name in enumerate(names):   # features → sibling scalar channels (BEV-colorable)
            channels[f"lidar_{name}"] = pts[:, 3 + i]
            specs.append(ChannelSpec(f"lidar_{name}", ChannelKind.SCALAR, pts.dtype, (None,)))
        session.set_source(ArraySource(specs, [channels]), keep_grid=keep)
        return view()

    # Web front mounted last (on "/"): the /api/* and /docs routes, registered before,
    # keep priority. `html=True` serves index.html at the root.
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    app.state.session = session
    return app


def serve(session_or_source, *, host: str = "127.0.0.1", port: int = 8000, labels=None):
    import uvicorn

    app = create_app(session_or_source, labels=labels)
    uvicorn.run(app, host=host, port=port)
    return app
