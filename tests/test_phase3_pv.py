"""Meshoid Grid / Derived Fields filters through the server manager (pvpython)."""

import os
import sys

import numpy as np
import pytest
from vtkmodules.util import numpy_support as ns

pvs = pytest.importorskip("paraview.simple")

from pvgizmo import io  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

PLUGINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin")


@pytest.fixture(scope="module", autouse=True)
def plugins():
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidGrid", "MeshoidFilters.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(PLUGINS, f), remote=False)


@pytest.fixture
def reader(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/single/snapshot_000.hdf5"])
    r.UpdatePipeline()
    yield r
    for s in list(pvs.GetSources().values()):
        pvs.Delete(s)


def out(f):
    f.UpdatePipeline()
    return f.GetClientSideObject().GetOutputDataObject(0)


def grid_values(img):
    a = img.GetPointData().GetArray(0)
    v = ns.vtk_to_numpy(a)
    nx, ny, nz = img.GetDimensions()
    return a.GetName(), v.reshape((nz, ny, nx) + v.shape[1:]).transpose((2, 1, 0) + tuple(range(3, v.ndim + 2)))


def gas(snapdir, fields=("Masses", "SmoothingLength", "Density", "Velocities")):
    return io.read_ptype(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5"), 0, list(fields))


def test_grid_deposit_matches_core_and_conserves(reader, snapdir):
    g = pvs.MeshoidGrid(Input=reader, AutoWindow=0, Center=[0, 0, 0], Size=60.0, Resolution=33)
    img = out(g)
    name, v = grid_values(img)
    assert name == "GridDensity_Masses" and img.GetDimensions() == (33, 33, 33)
    d = gas(snapdir)
    ref = mo.grid3d_deposit(d["Coordinates"], d["SmoothingLength"], d["Masses"], np.zeros(3), 60.0, 33)
    np.testing.assert_allclose(v, ref, rtol=1e-12)
    dx = 60.0 / 32
    np.testing.assert_allclose(img.GetOrigin(), [-30, -30, -30])
    np.testing.assert_allclose(img.GetSpacing(), [dx] * 3)
    inside = np.all(np.abs(d["Coordinates"]) + np.maximum(d["SmoothingLength"], dx)[:, None] < 30 - dx, axis=1)
    assert v.sum() * dx**3 >= d["Masses"][inside].sum() * (1 - 1e-6)
    assert v.sum() * dx**3 <= d["Masses"].sum() * (1 + 1e-6)


def test_contour_on_density_grid(reader):
    g = pvs.MeshoidGrid(Input=reader, Resolution=40)
    name, v = grid_values(out(g))
    c = pvs.Contour(Input=g, ContourBy=["POINTS", name], Isosurfaces=[float(np.percentile(v, 90))])
    c.UpdatePipeline()
    assert c.GetDataInformation().GetNumberOfCells() > 100


def test_velocity_grid_streamlines(reader, snapdir):
    g = pvs.MeshoidGrid(Input=reader, Mode="Interpolate (intensive field)", Field=["POINTS", "Velocities"],
                        Order=1, Resolution=24)
    img = out(g)
    name, v = grid_values(img)
    assert name == "Interp_Velocities" and v.shape == (24, 24, 24, 3)
    st = pvs.StreamTracer(Input=g, SeedType="Point Cloud", Vectors=["POINTS", name])
    st.UpdatePipeline()
    assert st.GetDataInformation().GetNumberOfPoints() > 10


def test_grid_interp_matches_core(reader, snapdir):
    from meshoid import Meshoid
    g = pvs.MeshoidGrid(Input=reader, Mode="Interpolate (intensive field)", Field=["POINTS", "Density"],
                        AutoWindow=0, Center=[0.1, 0, 0], Size=3.0, Resolution=12)
    _, v = grid_values(out(g))
    d = gas(snapdir)
    M = Meshoid(d["Coordinates"].astype(np.float64), d["Masses"].astype(np.float64))
    np.testing.assert_allclose(v, mo.grid3d_interp(M, d["Density"], [0.1, 0, 0], 3.0, 12), rtol=1e-12)


def test_derived_fields_gas(reader, snapdir):
    f = pvs.MeshoidDerivedFields(Input=reader, Field=["POINTS", "Velocities"], ComputeDivCurl=1,
                                 ComputeKernelAverage=1, ComputeDensity=1)
    o = out(f)
    assert o.GetClassName() == "vtkPartitionedDataSetCollection" and o.GetNumberOfPartitionedDataSets() == 3
    names = [o.GetMetaData(i).Get(o.NAME()) for i in range(3)]
    pd = o.GetPartitionedDataSet(names.index("Gas")).GetPartition(0)
    for k, nc in (("grad_Velocities", 9), ("div_Velocities", 1), ("curl_Velocities", 3),
                  ("KernelAverage_Velocities", 3), ("MeshoidDensity", 1), ("Density", 1)):
        a = pd.GetPointData().GetArray(k)
        assert a is not None and a.GetNumberOfComponents() == nc, k
    from meshoid import Meshoid
    d = gas(snapdir)
    M = Meshoid(d["Coordinates"].astype(np.float64), d["Masses"].astype(np.float64))
    np.testing.assert_allclose(ns.vtk_to_numpy(pd.GetPointData().GetArray("div_Velocities")),
                               M.Div(d["Velocities"].astype(np.float64)), rtol=1e-10, atol=1e-12)
    stars = o.GetPartitionedDataSet(names.index("Stars")).GetPartition(0)
    assert stars.GetPointData().GetArray("grad_Velocities") is None  # only the chosen block
    upstream = reader.GetClientSideObject().GetOutputDataObject(0)
    up_gas = upstream.GetPartitionedDataSet(0).GetPartition(0)
    assert up_gas.GetPointData().GetArray("grad_Velocities") is None  # input untouched


def test_derived_scalar_and_stars(reader):
    f = pvs.MeshoidDerivedFields(Input=reader, Block="Stars", Field=["POINTS", "Masses"],
                                 ComputeDivCurl=1, ComputeKernelRadius=1, ComputeDensity=1)
    o = out(f)
    names = [o.GetMetaData(i).Get(o.NAME()) for i in range(o.GetNumberOfPartitionedDataSets())]
    pd = o.GetPartitionedDataSet(names.index("Stars")).GetPartition(0)
    assert pd.GetPointData().GetArray("grad_Masses").GetNumberOfComponents() == 3
    assert pd.GetPointData().GetArray("div_Masses") is None  # scalar field: div/curl skipped
    h = ns.vtk_to_numpy(pd.GetPointData().GetArray("MeshoidKernelRadius"))
    assert np.all(h > 0) and np.all(ns.vtk_to_numpy(pd.GetPointData().GetArray("MeshoidDensity")) > 0)


def test_derived_on_plain_pointset(reader):
    m = pvs.MergeBlocks(Input=pvs.ExtractBlock(Input=reader, Selectors=["/Root/Gas"]))
    f = pvs.MeshoidDerivedFields(Input=m)
    o = out(f)
    assert o.IsA("vtkPointSet") and o.GetPointData().GetArray("grad_Density") is not None


def test_one_tree_shared_by_slice_grid_and_derived(reader):
    mod = next(m for m in sys.modules.values() if getattr(m, "MeshoidGrid", None) is not None
               and hasattr(m, "_CACHE"))
    mod._CACHE.clear()
    out(pvs.MeshoidSlice(Input=reader, Resolution=16))
    out(pvs.MeshoidGrid(Input=reader, Mode="Interpolate (intensive field)", Field=["POINTS", "Density"],
                        Resolution=8))
    out(pvs.MeshoidDerivedFields(Input=reader))
    assert len(mod._CACHE._d) == 1


def test_grid_log10_array(reader):
    g = pvs.MeshoidGrid(Input=reader, Resolution=16)
    img = out(g)
    pdat = img.GetPointData()
    assert pdat.GetNumberOfArrays() == 2 and pdat.GetArrayName(1) == "log10_GridDensity_Masses"
    lin, lg = (ns.vtk_to_numpy(pdat.GetArray(i)) for i in range(2))
    np.testing.assert_allclose(lg, mo.safe_log10(lin))
    assert np.all(np.isfinite(lg))
    g.AddLog10 = 0
    assert out(g).GetPointData().GetNumberOfArrays() == 1


def test_all_proxies_registered():
    """Invalid proxy XML (e.g. '<' in a Documentation string) fails silently: catch it here."""
    for name in ("GIZMOSnapshotReader", "MeshoidProjection", "MeshoidSlice", "MeshoidGrid", "MeshoidDerivedFields"):
        assert hasattr(pvs, name), name
