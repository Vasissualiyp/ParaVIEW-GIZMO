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
