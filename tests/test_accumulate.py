"""Headless tests: poses, accumulation by registration, point-label de-accumulation."""

import numpy as np

from splasher import ArraySource, ChannelKind, ChannelSpec
from splasher.core.accumulate import accumulate, window_indices
from splasher.core.poses import invert, pose_to_matrix, transform_points
from splasher.core.source import Frame
from splasher.core.target import PointTarget


def test_pose_to_matrix_forms():
    assert pose_to_matrix(np.eye(4)).shape == (4, 4)
    T = pose_to_matrix(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))  # identity quaternion
    assert np.allclose(T[:3, 3], [1, 2, 3])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_invert_and_transform():
    T = np.eye(4)
    T[:3, 3] = [5.0, 0.0, 0.0]
    pts = np.array([[1.0, 1.0, 0.0]])
    back = transform_points(transform_points(pts, T), invert(T))
    assert np.allclose(back[:, :3], pts)


def test_window_indices_clipped():
    assert window_indices(0, 2, 10) == [0, 1, 2]
    assert window_indices(5, 2, 10) == [3, 4, 5, 6, 7]
    assert window_indices(9, 3, 10) == [6, 7, 8, 9]


def _moving_source(n=5):
    specs = [
        ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3)),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]
    frames = []
    for t in range(n):
        # a point fixed in the WORLD at x=10; the ego moves forward by 1 per frame in x
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(t)
        world_x = 10.0
        pt_ego = np.array([[world_x - t, 0.0, 0.0]], dtype=np.float32)  # seen from the ego
        frames.append({"lidar": pt_ego, "pose": pose})
    return ArraySource(specs, frames)


def test_accumulate_registers_to_reference():
    src = _moving_source(5)
    ref = 2
    acc = accumulate(src, ref, window_indices(ref, 2, 5), ["lidar"], "pose")
    # all points (a single fixed world point) must land at the same place
    # in the ref frame's frame of reference: x = 10 - ref = 8
    assert len(acc.points) == 5
    assert np.allclose(acc.points[:, 0], 8.0, atol=1e-5)
    assert set(acc.frame_id.tolist()) == {0, 1, 2, 3, 4}


def test_accumulate_two_channels_chan_and_point_ids():
    specs = [
        ChannelSpec("a", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3)),
        ChannelSpec("b", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3)),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]
    frame = {
        "a": np.zeros((3, 3), np.float32),  # 3 points -> chan 0, point_id 0..2
        "b": np.ones((2, 3), np.float32),   # 2 points -> chan 1, point_id 3..4 (full concat)
        "pose": np.eye(4, dtype=np.float32),
    }
    src = ArraySource(specs, [frame])
    acc = accumulate(src, 0, [0], ["a", "b"], "pose")
    assert acc.counts[0] == 5
    assert acc.chan_id.tolist() == [0, 0, 0, 1, 1]
    assert acc.point_id.tolist() == [0, 1, 2, 3, 4]
    # visibility filter: keep only channel b (index 1)
    vis = acc.visible_mask([1])
    assert vis.tolist() == [False, False, False, True, True]


def test_point_features_maps_sibling_channels():
    from splasher.core.source import ordered_features, point_features

    specs = [
        ChannelSpec("velodyne_0", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3)),
        ChannelSpec("velodyne_0_intensity", ChannelKind.SCALAR, np.dtype("uint8"), (None,)),
        ChannelSpec("velodyne_0_range", ChannelKind.SCALAR, np.dtype("float32"), (None,)),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]
    # every sibling scalar becomes a named feature; unrelated kinds (pose) never pair.
    assert point_features(specs, ["velodyne_0"]) == {
        "velodyne_0": {"intensity": "velodyne_0_intensity", "range": "velodyne_0_range"}
    }
    # a cloud with no sibling scalar is simply absent from the mapping
    assert point_features(specs, ["lidar_other"]) == {}
    # column order: intensity first (keeps col 3), then alphabetical
    assert ordered_features({"range", "intensity", "azimuth"}) == ["intensity", "azimuth", "range"]


def test_accumulate_folds_features_into_trailing_columns():
    specs = [
        ChannelSpec("velodyne_0", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3)),
        ChannelSpec("velodyne_0_intensity", ChannelKind.SCALAR, np.dtype("uint8"), (None,)),
        ChannelSpec("velodyne_0_range", ChannelKind.SCALAR, np.dtype("float32"), (None,)),
    ]
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    inten = np.array([12, 200], np.uint8)                    # (N,) sidecars, KITTI/Tartan-style
    rng = np.array([7.5, 8.5], np.float32)
    src = ArraySource(specs, [{"velodyne_0": xyz, "velodyne_0_intensity": inten,
                               "velodyne_0_range": rng}])

    fmap = {"velodyne_0": {"intensity": "velodyne_0_intensity", "range": "velodyne_0_range"}}
    acc = accumulate(src, 0, [0], ["velodyne_0"], None, fmap, ["intensity", "range"])
    assert acc.points.shape == (2, 5)                        # [x, y, z, intensity, range]
    np.testing.assert_allclose(acc.points[:, :3], xyz)
    np.testing.assert_allclose(acc.points[:, 3], [12.0, 200.0])
    np.testing.assert_allclose(acc.points[:, 4], [7.5, 8.5])


def test_accumulate_missing_feature_is_nan():
    specs = [ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 3))]
    src = ArraySource(specs, [{"lidar": np.zeros((3, 3), np.float32)}])
    acc = accumulate(src, 0, [0], ["lidar"], None, None, ["intensity"])
    assert acc.points.shape == (3, 4)
    assert np.isnan(acc.points[:, 3]).all()
    # no feature layout requested → plain xyz
    assert accumulate(src, 0, [0], ["lidar"], None).points.shape == (3, 3)


def test_accumulate_skips_frames_missing_pose():
    # A real (apairo) source can yield frames lacking the pose channel after sync;
    # accumulate must skip those for registration instead of crashing.
    class _Src:
        def __len__(self): return 3
        def channels(self): return []
        def __getitem__(self, i):
            ch = {"lidar": np.full((2, 3), float(i), np.float32)}
            if i != 1:                       # frame 1 has no pose
                ch["pose"] = np.eye(4, dtype=np.float32)
            return Frame(channels=ch)

    acc = accumulate(_Src(), ref_idx=0, indices=[0, 1, 2], cloud_keys=["lidar"], pose_key="pose")
    assert set(acc.frame_id.tolist()) == {0, 2}   # frame 1 skipped, no KeyError


def test_decumul_via_apply_scatter():
    # the brush hits a point coming from 3 frames -> labels spread per frame
    src = _moving_source(5)
    ref = 2
    acc = accumulate(src, ref, window_indices(ref, 2, 5), ["lidar"], "pose")
    pt = PointTarget(ignore_id=0)
    frame_to_sel = {int(f): (acc.point_id[acc.frame_id == f], acc.counts[int(f)])
                    for f in np.unique(acc.frame_id)}
    assert pt.apply_scatter(ref, frame_to_sel, class_id=4)
    # each frame received its label on its point 0
    for f in range(5):
        assert pt.labels(f).tolist() == [4]
    pt.undo(ref)  # atomic undo under the reference frame: everything reverts
    for f in range(5):
        assert pt.labels(f).tolist() == [0]
