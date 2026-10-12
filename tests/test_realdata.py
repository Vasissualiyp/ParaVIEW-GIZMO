"""Opt-in checks against real GIZMO snapshots (never committed, never large).

    PV_GIZMO_REAL_SNAPSHOTS="a.hdf5:b.hdf5" pvpython -m pytest tests/test_realdata.py

Keep each file below a few hundred MB: every field is read twice.
"""

import os

import numpy as np
import pytest
from meshoid.load_from_snapshot import load_from_snapshot

from pvgizmo import io

FILES = [f for f in os.environ.get("PV_GIZMO_REAL_SNAPSHOTS", "").split(":") if f]
pytestmark = pytest.mark.skipif(not FILES, reason="set PV_GIZMO_REAL_SNAPSHOTS")


@pytest.fixture(params=FILES, ids=os.path.basename)
def snap(request):
    return request.param


def test_header_and_fields(snap):
    parts = io.snapshot_parts(snap)
    hdr = io.read_header(parts)
    fields = io.list_fields(parts)
    assert fields and 0 in fields
    # header counts can exceed what is stored (cutouts drop groups): list only stored types
    for t in fields:
        assert hdr["NumPart_Total"][t] >= io.read_ptype(parts, t, [])["Coordinates"].shape[0] > 0


@pytest.mark.parametrize("physical", [True, False])
def test_every_field_matches_load_from_snapshot(snap, physical):
    parts = io.snapshot_parts(snap)
    m = io._SNUM_RE.search(os.path.basename(snap))
    sdir, snum = os.path.dirname(snap), int(m.group(1))
    for t, fl in io.list_fields(parts).items():
        data = io.read_ptype(parts, t, list(fl), physical=physical)
        for k in fl:
            ref = load_from_snapshot(k, t, sdir, snum, units_to_physical=physical)
            if np.ndim(ref) == 0:  # PFH loader returns 0 for MassTable-only masses
                continue
            # We keep the file's float32 (PFH's loader upcasts to float64 before applying h),
            # so float32 denormals (< 1.2e-38) round differently: allow that absolute slack.
            np.testing.assert_allclose(data[k], ref, rtol=2e-7, atol=1e-37, err_msg=f"PartType{t}/{k}")


def test_reader_all_arrays(snap):
    pvs = pytest.importorskip("paraview.simple")
    if not hasattr(pvs, "GIZMOSnapshotReader"):
        pvs.LoadPlugin(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "plugin", "GIZMOReader.py"), remote=False)
    r = pvs.GIZMOSnapshotReader(FileNames=[snap])
    r.UpdatePipelineInformation()
    r.PointArrays = list(r.PointArrays.Available)
    r.UpdatePipeline()
    out = r.GetClientSideObject().GetOutputDataObject(0)
    parts = io.snapshot_parts(snap)
    fields = io.list_fields(parts)
    assert out.GetNumberOfPartitionedDataSets() == len(fields)
    for i, (t, fl) in enumerate(fields.items()):
        pd = out.GetPartitionedDataSet(i).GetPartition(0)
        assert out.GetMetaData(i).Get(out.NAME()) == io.PTYPE_NAMES[t]
        for k, (shape, _) in fl.items():
            if k == "Coordinates":
                continue
            a = pd.GetPointData().GetArray(k)
            assert a is not None, k
            assert a.GetNumberOfComponents() == int(np.prod(shape)) if shape else 1


def test_projection_matches_guac_pattern(snap):
    """Phase 2 acceptance: plugin map == GUAC's `Meshoid(pos, m, hsml).SurfaceDensity(M.m, ...)`."""
    import time
    from meshoid import Meshoid
    pvs = pytest.importorskip("paraview.simple")
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidProjection", "MeshoidFilters.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "plugin", f), remote=False)
    parts = io.snapshot_parts(snap)
    d = io.read_ptype(parts, 0, ["Masses", "SmoothingLength", "Density"])
    pos = d["Coordinates"].astype(np.float64)
    center = pos[np.argmax(d["Density"])]
    size = 4 * np.median(np.linalg.norm(pos - center, axis=1))
    res = 256
    M = Meshoid(pos, d["Masses"], d["SmoothingLength"])  # as in GUAC gal_viz.py & co.
    ref = M.SurfaceDensity(M.m, center=center, size=size, res=res)

    r = pvs.GIZMOSnapshotReader(FileNames=[snap])
    r.UpdatePipeline()
    p = pvs.MeshoidProjection(Input=r, AutoWindow=0, Center=list(center), Size=size, Resolution=res)
    t = time.time()
    p.UpdatePipeline()
    dt = time.time() - t
    img = p.GetClientSideObject().GetOutputDataObject(0)
    from vtkmodules.util import numpy_support as vns
    v = vns.vtk_to_numpy(img.GetPointData().GetArray(0)).reshape(res, res).T
    # ~1e-8 differences: the m12f cutouts are 6e-5 kpc across at ~1948 kpc, so centring before
    # vs. inside the deposition loses ~3e-9 of relative position precision in float64.
    np.testing.assert_allclose(v, ref, rtol=1e-6, atol=1e-9 * ref.max())
    print(f"\n{os.path.basename(snap)}: N_gas={len(pos)} res={res} projection {dt:.2f}s")
