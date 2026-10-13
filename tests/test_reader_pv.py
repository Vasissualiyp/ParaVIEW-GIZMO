"""Reader plugin through the ParaView server manager (run under pvpython)."""

import os

import numpy as np
import pytest
from vtkmodules.util import numpy_support as ns

pvs = pytest.importorskip("paraview.simple")

from pvgizmo import io  # noqa: E402

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin", "GIZMOReader.py")


@pytest.fixture(scope="module")
def pv():
    # Already present when the dev shell's PV_PLUGIN_PATH auto-loaded plugin/.
    if not hasattr(pvs, "GIZMOSnapshotReader"):
        pvs.LoadPlugin(PLUGIN, remote=False)
    return {"GIZMOSnapshotReader": pvs.GIZMOSnapshotReader}


def output(reader, t=None):
    """The reader's actual output object (builtin session), metadata intact."""
    reader.UpdatePipeline() if t is None else reader.UpdatePipeline(t)
    return reader.GetClientSideObject().GetOutputDataObject(0)


def blocks(reader):
    data = output(reader)
    out = {}
    for i in range(data.GetNumberOfPartitionedDataSets()):
        name = data.GetMetaData(i).Get(data.NAME())
        out[name] = data.GetPartitionedDataSet(i).GetPartition(0)
    return data, out


def arr(pd, name):
    return ns.vtk_to_numpy(pd.GetPointData().GetArray(name))


def test_single(pv, snapdir):
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/single/snapshot_000.hdf5"], Recenter=0)
    r.UpdatePipelineInformation()
    assert sorted(r.ParticleTypes.Available) == ["DarkMatter", "Gas", "Stars"]
    assert "ParticleIDs" in r.PointArrays.Available and "ParticleIDs" not in r.PointArrays
    assert "Metallicity" in r.PointArrays and "StellarFormationTime" in r.PointArrays  # starlight inputs
    data, b = blocks(r)
    assert sorted(b) == ["DarkMatter", "Gas", "Stars"]
    gas = b["Gas"]
    assert gas.GetNumberOfPoints() == 4000 and gas.GetNumberOfCells() == 1
    ref = io.read_ptype(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5"), 0, ["Density", "SmoothingLength"])
    np.testing.assert_array_equal(arr(gas, "Density"), ref["Density"])
    np.testing.assert_array_equal(ns.vtk_to_numpy(gas.GetPoints().GetData()), ref["Coordinates"])
    np.testing.assert_allclose(arr(gas, "SmoothingLengthVTK"), 0.5 * ref["SmoothingLength"])
    assert np.all(arr(b["DarkMatter"], "Masses") == 5e-3)  # from MassTable
    assert data.GetFieldData().GetArray("NumPart_Total").GetNumberOfTuples() == 6
    assert data.GetDataAssembly().GetNodeName(0) == "Root" and data.GetDataAssembly().GetNodeName(1) == "Gas"


def test_selection_stride_bbox(pv, snapdir):
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/single/snapshot_000.hdf5"], Recenter=0)
    r.UpdatePipelineInformation()
    r.ParticleTypes = ["Gas"]
    r.PointArrays = ["Metallicity", "ParticleIDs"]
    r.Stride = 4
    _, b = blocks(r)
    assert list(b) == ["Gas"]
    gas = b["Gas"]
    assert gas.GetNumberOfPoints() == 1000
    assert gas.GetPointData().GetArray("Metallicity").GetNumberOfComponents() == 11
    assert gas.GetPointData().GetArray("Density") is None
    np.testing.assert_array_equal(arr(gas, "ParticleIDs"), np.arange(1, 4001)[::4])
    r.Stride = 1
    r.UseBoundingBox = 1
    r.BoundingBox = [-1, 1, -1, 1, -1, 1]
    _, b = blocks(r)
    x = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    assert 0 < len(x) < 4000 and np.all(np.abs(x) <= 1)


def test_multipart_equals_single(pv, snapdir):
    parts = [f"{snapdir}/multi/snapdir_000/snapshot_000.{k}.hdf5" for k in range(3)]
    for files in ([parts[1]], parts):  # one part, or the whole group as ParaView passes it
        r = pv["GIZMOSnapshotReader"](FileNames=files)
        r.UpdatePipelineInformation()
        assert len(r.TimestepValues) == 0  # one snapshot, however many parts
        _, b = blocks(r)
        assert b["Gas"].GetNumberOfPoints() == 4000 and b["DarkMatter"].GetNumberOfPoints() == 2000


def test_time_series(pv, snapdir):
    files = [f"{snapdir}/series/snapshot_{i:03d}.hdf5" for i in (2, 0, 1)]  # unsorted on purpose
    r = pv["GIZMOSnapshotReader"](FileNames=files)
    r.UpdatePipelineInformation()
    np.testing.assert_allclose(list(r.TimestepValues), [0.0, 0.1, 0.2])
    for t in (0.0, 0.1, 0.2):
        data = output(r, t)
        assert data.GetFieldData().GetArray("Time").GetValue(0) == pytest.approx(t)


def test_cosmo_physical_toggle(pv, snapdir):
    f = f"{snapdir}/cosmo/snapshot_000.hdf5"
    r = pv["GIZMOSnapshotReader"](FileNames=[f], Recenter=0)  # absolute coords: test units only
    _, b = blocks(r)
    phys = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData()).copy()
    r.PhysicalUnits = 0
    data, b = blocks(r)
    code = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    np.testing.assert_allclose(phys, code * 0.5 / 0.7, rtol=1e-6)
    assert data.GetFieldData().GetArray("Cosmological").GetValue(0) == 1


def test_reexecution_does_not_accumulate_blocks(pv, snapdir):
    """The output object is reused; changing time/properties must not append blocks."""
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/series/snapshot_{i:03d}.hdf5" for i in range(3)])
    r.UpdatePipelineInformation()
    for t in (0.0, 0.2, 0.1):
        assert output(r, t).GetNumberOfPartitionedDataSets() == 3
    r.ParticleTypes = ["Gas"]
    assert output(r, 0.1).GetNumberOfPartitionedDataSets() == 1
    e = pvs.ExtractBlock(Input=r, Selectors=["/Root/Gas"])
    e.UpdatePipeline(0.2)
    assert e.GetDataInformation().GetNumberOfPoints() == 4000


@pytest.mark.parametrize("prerun", [False, True])
def test_extract_block_selector_stable(pv, snapdir, prerun):
    """/Root/<type> must work whether ExtractBlock picks the Hierarchy or our Assembly."""
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/single/snapshot_000.hdf5"])
    if prerun:
        r.UpdatePipeline()
    for name, n in (("Gas", 4000), ("Stars", 500), ("DarkMatter", 2000)):
        e = pvs.ExtractBlock(Input=r, Selectors=[f"/Root/{name}"])
        e.UpdatePipeline()
        assert e.GetDataInformation().GetNumberOfPoints() == n, (name, e.Assembly)


def test_recenter_zoom_survives_float32(pv, snapdir):
    """A 1e-4 cloud at ~2000 collapses in float32 (what the GPU sees) unless recentred."""
    import snapgen
    f = f"{snapdir}/zoom/snapshot_000.hdf5"
    parts = io.snapshot_parts(f)
    ref = io.read_ptype(parts, 0, ["Density"])
    peak = ref["Coordinates"][np.argmax(ref["Density"])]
    r = pv["GIZMOSnapshotReader"](FileNames=[f], Recenter=0)
    _, b = blocks(r)
    absolute = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    assert len(np.unique(absolute.astype(np.float32), axis=0)) < 100  # collapsed
    r.Recenter = 1
    data, b = blocks(r)
    x = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    np.testing.assert_array_equal(x, ref["Coordinates"] - peak)  # exact, done in float64
    assert len(np.unique(x.astype(np.float32), axis=0)) == 4000
    off = data.GetFieldData().GetArray("CenterOffset")
    np.testing.assert_array_equal([off.GetValue(i) for i in range(3)], peak)
    assert np.linalg.norm(peak - snapgen.ZOOM_CENTER) < 1e-4
    r.Recenter = 2
    r.Center = list(snapgen.ZOOM_CENTER)
    _, b = blocks(r)
    np.testing.assert_allclose(ns.vtk_to_numpy(b["Gas"].GetPoints().GetData()),
                               ref["Coordinates"] - snapgen.ZOOM_CENTER, rtol=0, atol=1e-12)


def test_recenter_is_opt_in(pv, snapdir):
    import GIZMOReader
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/zoom/snapshot_000.hdf5"])
    assert r.Recenter == "None"  # enum properties read back as their label
    data, b = blocks(r)
    x = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    assert GIZMOReader.float32_collapses(x)  # this is what triggers the reader's warning
    assert [data.GetFieldData().GetArray("CenterOffset").GetValue(i) for i in range(3)] == [0, 0, 0]
    _, b = blocks(pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/single/snapshot_000.hdf5"]))
    assert not GIZMOReader.float32_collapses(ns.vtk_to_numpy(b["Gas"].GetPoints().GetData()))


def test_bbox_in_recentred_frame(pv, snapdir):
    r = pv["GIZMOSnapshotReader"](FileNames=[f"{snapdir}/zoom/snapshot_000.hdf5"], Recenter=1, UseBoundingBox=1,
                                  BoundingBox=[-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6])
    _, b = blocks(r)
    x = ns.vtk_to_numpy(b["Gas"].GetPoints().GetData())
    assert 0 < len(x) < 4000 and np.all(np.abs(x) <= 1e-6)
