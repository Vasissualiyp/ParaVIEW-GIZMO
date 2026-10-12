import os

import numpy as np
import pytest
from meshoid.load_from_snapshot import load_from_snapshot

from pvgizmo import io

CASES = [("single", "single/snapshot_000.hdf5"),
         ("multi", "multi/snapdir_000/snapshot_000.1.hdf5"),
         ("cosmo", "cosmo/snapshot_000.hdf5")]


def test_parts_discovery(snapdir):
    assert io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5") == [f"{snapdir}/single/snapshot_000.hdf5"]
    want = [f"{snapdir}/multi/snapdir_000/snapshot_000.{k}.hdf5" for k in range(3)]
    for p in (want[0], want[2], f"{snapdir}/multi/snapdir_000"):
        assert io.snapshot_parts(p) == want
    assert io.snapshot_key(want[0]) == io.snapshot_key(want[2])
    assert io.snapshot_number(want[1]) == 0
    assert io.snapshot_number(f"{snapdir}/series/snapshot_002.hdf5") == 2


@pytest.mark.parametrize("name,rel", CASES)
@pytest.mark.parametrize("physical", [True, False])
def test_matches_load_from_snapshot(snapdir, name, rel, physical):
    """Every dataset of every type equals PFH's reference loader."""
    parts = io.snapshot_parts(os.path.join(snapdir, rel))
    sdir = os.path.join(snapdir, name)
    fields = io.list_fields(parts)
    assert sorted(fields) == [0, 1, 4]
    for t, fl in fields.items():
        names = [k for k in fl if not (k == "Masses" and t == 1)]
        data = io.read_ptype(parts, t, names, physical=physical)
        for k in names:
            ref = load_from_snapshot(k, t, sdir, 0, units_to_physical=physical)
            np.testing.assert_allclose(data[k], ref, rtol=1e-6, err_msg=f"{name} PartType{t}/{k}")


def test_cosmo_units(snapdir):
    parts = io.snapshot_parts(f"{snapdir}/cosmo/snapshot_000.hdf5")
    hdr = io.read_header(parts)
    assert io.is_cosmological(hdr)
    code = io.read_ptype(parts, 0, ["Density", "Masses"], physical=False)
    phys = io.read_ptype(parts, 0, ["Density", "Masses"], physical=True)
    a, h = 0.5, 0.7
    np.testing.assert_allclose(phys["Coordinates"], code["Coordinates"] * a / h, rtol=1e-6)
    np.testing.assert_allclose(phys["Density"], code["Density"] * h**2 / a**3, rtol=1e-6)
    np.testing.assert_allclose(phys["Masses"], code["Masses"] / h, rtol=1e-6)


def test_masstable_masses(snapdir):
    parts = io.snapshot_parts(f"{snapdir}/multi/snapdir_000")
    assert "Masses" in io.list_fields(parts)[1]
    d = io.read_ptype(parts, 1, ["Masses"])
    assert d["Masses"].shape == (2000,) and np.all(d["Masses"] == 5e-3)


def test_header(snapdir):
    hdr = io.read_header(io.snapshot_parts(f"{snapdir}/multi/snapdir_000"))
    assert list(hdr["NumPart_Total"]) == [4000, 2000, 0, 0, 500, 0]
    assert hdr["NumFilesPerSnapshot"] == 3


@pytest.mark.parametrize("stride", [1, 3, 7])
def test_stride_consistent_across_parts(snapdir, stride):
    single = io.read_ptype(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5"), 0, ["ParticleIDs"], stride=stride)
    multi = io.read_ptype(io.snapshot_parts(f"{snapdir}/multi/snapdir_000"), 0, ["ParticleIDs"], stride=stride)
    np.testing.assert_array_equal(single["ParticleIDs"], multi["ParticleIDs"])
    np.testing.assert_array_equal(single["ParticleIDs"], np.arange(1, 4001)[::stride])


def test_bbox(snapdir):
    parts = io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5")
    full = io.read_ptype(parts, 0, ["Masses"])
    box = (-1, 1, -0.5, 0.5, -2, 2)
    d = io.read_ptype(parts, 0, ["Masses"], bbox=box)
    x = full["Coordinates"]
    want = (abs(x[:, 0]) <= 1) & (abs(x[:, 1]) <= 0.5) & (abs(x[:, 2]) <= 2)
    assert 0 < len(d["Masses"]) == want.sum()
    np.testing.assert_array_equal(d["Coordinates"], x[want])


def test_missing_type_is_empty(snapdir):
    assert io.read_ptype(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5"), 2, []) == {}


def test_center_and_density_peak(snapdir):
    parts = io.snapshot_parts(f"{snapdir}/zoom/snapshot_000.hdf5")
    d = io.read_ptype(parts, 0, ["Density"])
    assert d["Coordinates"].dtype == np.float64
    peak = io.density_peak(parts)
    np.testing.assert_array_equal(peak, d["Coordinates"][np.argmax(d["Density"])])
    c = io.read_ptype(parts, 0, [], center=peak)
    np.testing.assert_array_equal(c["Coordinates"], d["Coordinates"] - peak)
    assert io.density_peak(io.snapshot_parts(f"{snapdir}/single/snapshot_000.hdf5")) is not None
