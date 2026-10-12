"""Meshoid Projection / Slice filters through the ParaView server manager (pvpython)."""

import os

import numpy as np
import pytest
from vtkmodules.util import numpy_support as ns

pvs = pytest.importorskip("paraview.simple")

from pvgizmo import io  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

PLUGINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin")


@pytest.fixture(scope="module", autouse=True)
def plugins():
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidProjection", "MeshoidFilters.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(PLUGINS, f), remote=False)


@pytest.fixture
def reader(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/single/snapshot_000.hdf5"])
    r.UpdatePipeline()  # filters' array domains need the reader to have run (as in the GUI)
    yield r
    for p in list(pvs.GetSources().values()):
        pvs.Delete(p)


def image(filt, t=None):
    filt.UpdatePipeline() if t is None else filt.UpdatePipeline(t)
    img = filt.GetClientSideObject().GetOutputDataObject(0)
    a = img.GetPointData().GetArray(0)
    v = ns.vtk_to_numpy(a)
    nx, ny, _ = img.GetDimensions()
    v = v.reshape((ny, nx) + v.shape[1:]).swapaxes(0, 1)  # VTK x-fastest -> [i, j]
    return img, a.GetName(), v


def gas(snapdir, fields=("Masses", "SmoothingLength", "Density", "Velocities")):
    return io.read_ptype(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5"), 0, list(fields))


def test_defaults(reader):
    p = pvs.MeshoidProjection(Input=reader)
    assert list(p.Field) == ["POINTS", "Masses"] and p.Mode == "Surface density" and p.Block == "Gas"
    assert list(pvs.MeshoidSlice(Input=reader).Field) == ["POINTS", "Density"]


def test_projection_matches_core_and_geometry(reader, snapdir):
    n, up = [1.0, 2.0, 0.5], [0.0, 0.0, 1.0]
    p = pvs.MeshoidProjection(Input=reader, Normal=n, ViewUp=up, AutoWindow=0, Center=[0.1, 0, -0.2],
                              Size=4.0, Resolution=48)
    img, name, v = image(p)
    assert name == "SurfaceDensity_Masses" and img.GetDimensions() == (48, 48, 1)
    d = gas(snapdir)
    ref = mo.project("SurfaceDensity", d["Coordinates"], d["SmoothingLength"], [0.1, 0, -0.2], 4.0, 48,
                     normal=n, view_up=up, field=d["Masses"])
    np.testing.assert_allclose(v, ref, rtol=1e-12)
    pts = mo.slice_points([0.1, 0, -0.2], 4.0, 48, n, up)
    for i, j in [(0, 0), (47, 3), (20, 31)]:
        np.testing.assert_allclose(img.GetPoint(img.ComputePointId([i, j, 0])), pts[i * 48 + j], atol=1e-12)


@pytest.mark.parametrize("mode,field,ncomp", [("Line-of-sight integral", "Density", 1),
                                              ("Mass-weighted average", "Velocities", 3),
                                              ("Projected average", "InternalEnergy", 1)])
def test_other_modes(reader, snapdir, mode, field, ncomp):
    p = pvs.MeshoidProjection(Input=reader, Mode=mode, Field=["POINTS", field], Resolution=32)
    img, name, v = image(p)
    assert v.shape[:2] == (32, 32) and (v.shape[2] if v.ndim == 3 else 1) == ncomp
    assert np.isfinite(v).mean() > 0.5
    if mode == "Line-of-sight integral":  # volumes come from the snapshot: m / rho
        d = gas(snapdir)
        c, size = mo.auto_extent(d["Coordinates"])
        ref = mo.project("LineIntegral", d["Coordinates"], d["SmoothingLength"], c, size, 32,
                         field=d["Density"], volume=d["Masses"] / d["Density"])
        np.testing.assert_allclose(v, ref, rtol=1e-12)


def test_stars_without_kernel_radius(reader):
    p = pvs.MeshoidProjection(Input=reader, Block="Stars", Resolution=32)
    _, _, v = image(p)
    assert v.sum() > 0  # kernel radii computed by meshoid


def test_slice_matches_core(reader, snapdir):
    s = pvs.MeshoidSlice(Input=reader, Normal=[0, 1, 1], ViewUp=[1, 0, 0], Order=1, Resolution=40,
                         AutoWindow=0, Center=[0, 0, 0], Size=2.0)
    _, name, v = image(s)
    assert name == "Slice_Density"
    from meshoid import Meshoid
    d = gas(snapdir)
    M = Meshoid(d["Coordinates"].astype(np.float64), d["Masses"].astype(np.float64),
                kernel_radius=d["SmoothingLength"].astype(np.float64))
    ref = mo.slice_field(M, d["Density"], [0, 0, 0], 2.0, 40, [0, 1, 1], [1, 0, 0], order=1)
    np.testing.assert_allclose(v, ref, rtol=1e-10)


def test_projection_follows_time(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/series/snapshot_{i:03d}.hdf5" for i in range(3)])
    r.UpdatePipeline(0.0)
    p = pvs.MeshoidProjection(Input=r, AutoWindow=0, Size=6.0, Resolution=32)
    _, _, v0 = image(p, 0.0)
    _, _, v2 = image(p, 0.2)
    assert not np.allclose(v0, v2)  # the series expands with time
    assert v2.max() < v0.max()


def test_errors_do_not_crash(reader):
    p = pvs.MeshoidProjection(Input=reader, Block="NoSuchType")
    p.UpdatePipeline()
    assert p.GetDataInformation().GetNumberOfPoints() == 0


def test_project_along_camera(reader):
    from pvgizmo import pvhelpers
    view = pvs.CreateRenderView()
    cam = view.GetActiveCamera()
    cam.SetFocalPoint(0.5, 0, 0)
    cam.SetPosition(0.5, -10, 0)
    cam.SetViewUp(0, 0, 1)
    cam.SetViewAngle(30)
    p = pvs.MeshoidProjection(Input=reader, Resolution=16)
    pvhelpers.project_along_camera(p, view, render=False)
    np.testing.assert_allclose(p.Normal, [0, -1, 0])
    np.testing.assert_allclose(p.Center, [0.5, 0, 0])
    assert p.Size == pytest.approx(2 * 10 * np.tan(np.radians(15)))
    img, _, _ = image(p)
    b = img.GetBounds()  # image lies in the plane y = 0, facing the camera
    assert b[2] == pytest.approx(0, abs=1e-12) and b[3] == pytest.approx(0, abs=1e-12)


def test_field_placeholder_falls_back_to_default(snapdir):
    """Created before the input ran (scripts): the domain gives '0'; use the default array."""
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/single/snapshot_000.hdf5"])  # not run yet
    p = pvs.MeshoidProjection(Input=r, Resolution=16)
    s = pvs.MeshoidSlice(Input=r, Resolution=16)
    for f, want in ((p, "SurfaceDensity_Masses"), (s, "Slice_Density")):
        f.UpdatePipeline()
        assert f.PointData.keys() == [want]
