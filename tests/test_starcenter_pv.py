"""Star-centred movies and dense-gas volumes (pvpython): Recenter = Most massive star,
CenterParticleID tracking, star_camera/zoom, and the camera-following Meshoid Grid with
scaled_log10 opacity."""

import os

import numpy as np
import pytest
from vtkmodules.util import numpy_support as ns

pvs = pytest.importorskip("paraview.simple")

import snapgen  # noqa: E402
from pvgizmo import io, live, pvhelpers  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGINS = os.path.join(os.path.dirname(HERE), "plugin")
SHIFTS = ([0.0, 0.0, 0.0], [5.0, -3.0, 1.0], [10.0, 2.0, -4.0])  # the cluster drifts


@pytest.fixture(scope="module", autouse=True)
def plugins():
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidProjection", "MeshoidFilters.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(PLUGINS, f), remote=False)
    yield
    live.stop_all()
    for s in list(pvs.GetSources().values()):
        pvs.Delete(s)


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    """Three sink snapshots; the cluster moves, and in the last one a different sink is the
    most massive."""
    out = tmp_path_factory.mktemp("starseries")
    files, heaviest = [], []
    for i, shift in enumerate(SHIFTS):
        parts, mtab = snapgen.make_sinks(ngas=1500, nsink=12, seed=6)
        for d in parts.values():
            d["Coordinates"] = (d["Coordinates"] + np.float32(shift)).astype(np.float32)
        m = parts[5]["Masses"]
        if i == 2:  # another sink overtakes the previous heaviest
            m[np.argsort(m)[-2]] = 2 * m.max()
        heaviest.append(int(np.argmax(m)))
        base = f"{out}/snapshot_{i:03d}"
        snapgen.write_snapshot(base, parts, mtab, time=0.1 * i)
        files.append(base + ".hdf5")
    return files, heaviest


def test_star_center_io(series):
    files, heaviest = series
    for f, k in zip(files, heaviest):
        parts = io.snapshot_parts(f)
        d = io.read_ptype(parts, 5, ["Masses", "ParticleIDs"])
        pos, pid, t = io.star_center(parts)
        assert t == 5 and pid == d["ParticleIDs"][k]
        assert np.allclose(pos, d["Coordinates"][k])
    first = io.star_center(io.snapshot_parts(files[0]))[1]
    pos, pid, _ = io.star_center(io.snapshot_parts(files[2]), particle_id=first)
    assert pid == first  # tracked by ID although no longer the heaviest
    assert io.star_center(io.snapshot_parts(files[0]), particle_id=10**9)[1] == first  # absent: fallback


def _star_frame(r, t, pid=None):
    r.UpdatePipeline(t)
    out = r.GetClientSideObject().GetOutputDataObject(0)
    sinks = pvhelpers_block(out, "Type5")
    pos = ns.vtk_to_numpy(sinks.GetPoints().GetData())
    m = ns.vtk_to_numpy(sinks.GetPointData().GetArray("Masses"))
    used = out.GetFieldData().GetArray("CenterParticleID").GetValue(0)
    return pos, m, used


def pvhelpers_block(out, name):
    for i in range(out.GetNumberOfPartitionedDataSets()):
        if out.GetMetaData(i).Get(out.NAME()) == name:
            return out.GetPartitionedDataSet(i).GetPartition(0)
    raise KeyError(name)


def test_reader_centres_each_snapshot(series):
    files, _ = series
    r = pvs.GIZMOSnapshotReader(FileNames=files)
    pvhelpers.center_on_star(r)
    times = r.TimestepValues
    assert len(times) == 3
    ids = []
    for t in times:
        pos, m, used = _star_frame(r, t)
        assert np.allclose(pos[np.argmax(m)], 0.0, atol=1e-6)  # heaviest sink at the origin
        ids.append(used)
    assert ids[0] == ids[1] != ids[2]  # the most massive star changed in the last snapshot
    r.CenterParticleID = int(ids[0])  # follow the first one instead
    pos, m, used = _star_frame(r, times[2])
    assert used == ids[0]
    k = np.flatnonzero(np.all(np.abs(pos) < 1e-6, axis=1))
    assert k.size == 1 and m[k[0]] < m.max()  # centred on the tracked, now lighter, star
    pvs.Delete(r)


def test_star_camera_and_zoom(series):
    files, _ = series
    r = pvhelpers.center_on_star(pvs.GIZMOSnapshotReader(FileNames=files[:1]))
    view = pvs.CreateRenderView()
    view.ViewSize = [400, 300]
    pvs.Show(r, view)
    pvs.ResetCamera(view)
    view.CameraFocalPoint = [3.0, 1.0, 0.0]  # what Reset Camera gives in a zoom run: off the star
    pvhelpers.star_camera(view, size=2.0)
    assert list(view.CameraFocalPoint) == [0, 0, 0] and list(view.CenterOfRotation) == [0, 0, 0]
    assert view.CameraParallelProjection == 1
    assert live.view_window(view)[3] == pytest.approx(2.0)
    for _ in range(3):
        pvhelpers.zoom(10.0, view)
    assert live.view_window(view)[3] == pytest.approx(2e-3)  # 1000x in, still on the star
    assert list(view.CameraFocalPoint) == [0, 0, 0]
    pvs.Delete(view)
    pvs.Delete(r)


def test_scaled_log10():
    v = np.array([0.0, 1e-6, 1e-4, 1e-2, 1.0, 1.0])
    s = mo.scaled_log10(v, decades=4.0, top_percentile=100)
    assert s.dtype == np.float32
    assert np.allclose(s, [0, 0, 0, 0.5, 1, 1])
    fixed = mo.scaled_log10(v, levels=(-5.0, -1.0))  # absolute levels: the same in any window
    assert np.allclose(fixed, [0, 0, 0.25, 0.75, 1, 1])
    assert np.allclose(mo.scaled_log10(v * 1e-3, levels=(-5.0, -1.0))[4], 0.5)


def test_dense_volume_follows_zoom(series):
    files, _ = series
    r = pvhelpers.center_on_star(pvs.GIZMOSnapshotReader(FileNames=files[:1]))
    view = pvs.CreateRenderView()
    view.ViewSize = [300, 300]
    pvhelpers.star_camera(view, size=4.0)
    g = pvhelpers.add_dense_volume(r, view, resolution=32)
    assert live.following(g)
    name = next(n for n in g.PointData.keys() if n.startswith("scaled_log10_"))
    lo, hi = g.PointData[name].GetRange()
    assert 0.0 <= lo < 0.5 and hi == pytest.approx(1.0)  # a dense core within a 0-1 scale
    rep = pvs.GetDisplayProperties(g, view)
    assert rep.Representation == "Volume"
    assert g.Size == pytest.approx(4.0) and list(g.Center) == pytest.approx([0, 0, 0], abs=1e-9)
    unit = rep.ScalarOpacityUnitDistance
    assert unit == pytest.approx(0.1 * 4.0)
    mtime = g.GetClientSideObject().GetMTime()
    view.GetActiveCamera().Azimuth(40)  # rotation: same window -> no re-grid
    pvs.Render(view)
    assert g.GetClientSideObject().GetMTime() == mtime
    pvhelpers.zoom(8.0, view)  # zoom: re-gridded on the smaller window, opacity length follows
    assert g.Size == pytest.approx(0.5)
    assert rep.ScalarOpacityUnitDistance == pytest.approx(unit / 8)
    g.UpdatePipeline()
    out = g.GetClientSideObject().GetOutputDataObject(0)
    assert out.GetSpacing()[0] == pytest.approx(0.5 / 31)
    a = ns.vtk_to_numpy(out.GetPointData().GetArray(name))
    g.ScaleRange = [-30.0, -29.0]  # far below any density: everything at 1
    g.UpdatePipeline()
    b = ns.vtk_to_numpy(g.GetClientSideObject().GetOutputDataObject(0).GetPointData().GetArray(name))
    assert not np.allclose(a, b) and np.median(b) == pytest.approx(1.0)
    pvs.Delete(view)
