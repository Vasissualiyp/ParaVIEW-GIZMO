"""Phase 6 filters through the server manager (pvpython): Starlight Image, Meshoid Composite,
Star Properties; colour-map presets."""

import os
import subprocess
import sys

import numpy as np
import pytest
from vtkmodules.util import numpy_support as ns

pvs = pytest.importorskip("paraview.simple")

from pvgizmo import io, live, looks, stars  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGINS = os.path.join(os.path.dirname(HERE), "plugin")


@pytest.fixture(scope="module", autouse=True)
def plugins():
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidProjection", "MeshoidFilters.py"),
                    ("GIZMOStarlightImage", "SpaceLook.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(PLUGINS, f), remote=False)


@pytest.fixture
def galaxy(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/galaxy/snapshot_600.hdf5"])
    r.UpdatePipeline()
    yield r
    live.stop_all()
    for s in list(pvs.GetSources().values()):
        pvs.Delete(s)


def maps(filt, name):
    filt.UpdatePipeline()
    o = filt.GetClientSideObject().GetOutputDataObject(0)
    e = o.GetExtent()
    nx, ny = e[1] - e[0] + 1, e[3] - e[2] + 1
    a = ns.vtk_to_numpy(o.GetPointData().GetArray(name))
    return a.reshape((ny, nx) + a.shape[1:]).swapaxes(0, 1)  # [i, j]


def read(snapdir, name, ptype, fields):
    parts = io.snapshot_parts(f"{snapdir}/{name}")
    return io.read_ptype(parts, ptype, fields), io.read_header(parts)


def test_proxies_registered():
    for name in ("GIZMOStarlightImage", "MeshoidComposite", "GIZMOStarProperties"):
        assert hasattr(pvs, name), name


def test_starlight_matches_core(galaxy, snapdir):
    n, up = [0.0, 0.3, 1.0], [0.0, 1.0, 0.0]
    s = pvs.GIZMOStarlightImage(Input=galaxy, Normal=n, ViewUp=up, AutoWindow=0, Center=[0.5, 0, 0],
                                Size=20.0, Resolution=64)
    rgb = maps(s, "RGB")
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    assert s.PointData.keys() == ["RGB", "Starlight_u", "Starlight_g", "Starlight_r"]
    st, hdr = read(snapdir, "galaxy/snapshot_600.hdf5", 4, ["Masses", "StellarFormationTime", "Metallicity"])
    gas, _ = read(snapdir, "galaxy/snapshot_600.hdf5", 0, ["Masses", "SmoothingLength", "Metallicity"])
    meta = {k: hdr[k] for k in ("UnitMass_In_CGS", "UnitLength_In_CGS", "UnitVelocity_In_CGS", "HubbleParam",
                                "Omega_Matter", "Omega_Lambda", "Time")}
    meta.update(Cosmological=1, PhysicalUnits=1)
    L, _ = stars.star_luminosities(("u", "g", "r"), st["Masses"], meta, sft=st["StellarFormationTime"],
                                   z=st["Metallicity"][:, 0])
    from meshoid import Meshoid
    hs = Meshoid(st["Coordinates"].astype(np.float64), st["Masses"].astype(np.float64)).SmoothingLength()
    hs = 1.0 / (1.0 / (3.0 * hs) + 1.0 / (0.01 * 20.0))
    x = np.r_[st["Coordinates"], gas["Coordinates"]]
    h = np.r_[hs, gas["SmoothingLength"]]
    mext = np.r_[np.zeros(len(hs)), gas["Masses"] * gas["Metallicity"][:, 0] / 0.02]
    lum = np.r_[L, np.zeros((len(mext) - len(hs), 3))]
    kappa = [stars.kappa_code(b, meta) * k for b, k in zip(("u", "g", "r"), stars.KAPPA_TWEAK)]
    ref, _ = stars.raytrace(x, h, mext, lum, [0.5, 0, 0], mo.view_basis(n, up), 20.0, 64, kappa)
    for k, b in enumerate("ugr"):
        np.testing.assert_allclose(maps(s, f"Starlight_{b}"), ref[..., k], rtol=1e-6, atol=1e-9 * ref.max())
    maxden, minden = stars.threeband_levels(ref)
    np.testing.assert_array_equal(rgb, stars.threeband_rgb(ref, maxden, minden))
    assert s.FieldData["BrightnessLevels"].GetRange() == pytest.approx((minden, maxden))


def test_starlight_extinction_dims_and_reddens_edge_on(galaxy):
    kw = dict(Normal=[0, -1, 0], ViewUp=[0, 0, 1], AutoWindow=0, Size=20.0, Resolution=96)
    dusty = pvs.GIZMOStarlightImage(Input=galaxy, **kw)
    clear = pvs.GIZMOStarlightImage(Input=galaxy, Extinction=0, **kw)
    tot = {}
    for name, f in (("dusty", dusty), ("clear", clear)):
        tot[name] = np.array([maps(f, f"Starlight_{b}").sum() for b in "ugr"])
    frac = tot["dusty"] / tot["clear"]
    assert np.all(frac < 0.95), frac  # the disc's dust lane absorbs
    assert frac[0] < frac[1] < frac[2], frac  # more in the blue: reddening
    face = pvs.GIZMOStarlightImage(Input=galaxy, Normal=[0, 0, 1], AutoWindow=0, Size=20.0, Resolution=96)
    face_clear = pvs.GIZMOStarlightImage(Input=galaxy, Normal=[0, 0, 1], AutoWindow=0, Size=20.0,
                                         Resolution=96, Extinction=0)
    f_face = maps(face, "Starlight_u").sum() / maps(face_clear, "Starlight_u").sum()
    assert f_face > frac[0]  # face-on sightlines cross less dust than edge-on ones


def test_starlight_young_ring_is_blue(galaxy):
    s = pvs.GIZMOStarlightImage(Input=galaxy, AutoWindow=0, Size=20.0, Resolution=100, Extinction=0)
    rgb = maps(s, "RGB").astype(float)
    g = np.linspace(-10, 10, 100)
    r = np.hypot(*np.meshgrid(g, g, indexing="ij"))
    ring, bulge = (r > 4.3) & (r < 5.7), r < 1.0

    def blue(m):  # blue - red of the region's lit pixels (stars, not the black floor of 2, 2, 2)
        v = rgb[m]
        top = v[v.sum(axis=1) > 30]
        assert len(top) > 20
        return (top[:, 2] - top[:, 0]).mean()
    assert blue(ring) > 10 and blue(ring) > blue(bulge) + 10, (blue(ring), blue(bulge))


def test_starlight_sinks(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/sinks/snapshot_010.hdf5"])
    s = pvs.GIZMOStarlightImage(Input=r, StarBlocks="Type5", Resolution=48)
    rgb = maps(s, "RGB")
    assert rgb.max() > 200 and (rgb.sum(axis=2) > 30).mean() > 0.001
    s.StarBlocks = "Nope"
    s.UpdatePipeline()
    assert s.GetDataInformation().GetNumberOfPoints() == 0  # error reported, no crash
    for p in list(pvs.GetSources().values()):
        pvs.Delete(p)


def test_composite_matches_projections(galaxy):
    c = pvs.MeshoidComposite(Input=galaxy, AutoWindow=0, Center=[0, 0, 0], Size=20.0, Resolution=48)
    assert list(c.Field2) == ["POINTS", "Temperature"] and c.Mode2 == "Mass-weighted average"
    rgb = maps(c, "RGB")
    sig, tw = maps(c, "SurfaceDensity_Masses"), maps(c, "MassWeightedAverage_Temperature")
    p1 = pvs.MeshoidProjection(Input=galaxy, AutoWindow=0, Size=20.0, Resolution=48)
    p2 = pvs.MeshoidProjection(Input=galaxy, Field=["POINTS", "Temperature"], Mode="Mass-weighted average",
                               AutoWindow=0, Size=20.0, Resolution=48)
    np.testing.assert_allclose(sig, maps(p1, "SurfaceDensity_Masses"), rtol=1e-12)
    np.testing.assert_allclose(tw, maps(p2, "MassWeightedAverage_Temperature"), rtol=1e-12)
    ref, lr, hr = looks.hsv_composite(sig, tw, "magma", None, None, True, True)
    np.testing.assert_array_equal(rgb, ref)
    assert c.FieldData["LightnessRange"].GetRange() == pytest.approx(lr)
    c.LightnessRange = [lr[0], lr[1] * 0.1]  # explicit range: brighter
    brighter = maps(c, "RGB")
    assert brighter.astype(int).sum() > rgb.astype(int).sum()
    np.testing.assert_array_equal(brighter, looks.hsv_composite(sig, tw, "magma", (lr[0], lr[1] * 0.1), None,
                                                                True, True)[0])
    c.Colormap = "viridis"
    np.testing.assert_array_equal(maps(c, "RGB"), looks.hsv_composite(sig, tw, "viridis", (lr[0], lr[1] * 0.1),
                                                                      None, True, True)[0])


def test_star_properties(galaxy):
    sp = pvs.GIZMOStarProperties(Input=galaxy)
    sp.UpdatePipeline()
    o = sp.GetClientSideObject().GetOutputDataObject(0)
    names = [o.GetMetaData(i).Get(o.NAME()) for i in range(o.GetNumberOfPartitionedDataSets())]
    pd = o.GetPartitionedDataSet(names.index("Stars")).GetPartition(0)
    get = lambda k: ns.vtk_to_numpy(pd.GetPointData().GetArray(k))  # noqa: E731
    age, rgb, size = get("StellarAge"), get("StarRGB").astype(int), get("StarSize")
    young = age < 0.2
    assert young.sum() == 1500 and age[~young].min() > 9
    assert (rgb[young, 2] - rgb[young, 0]).mean() > (rgb[~young, 2] - rgb[~young, 0]).mean() + 50
    assert np.all(np.diff(np.sort(get("Luminosity"))) >= 0) and get("Luminosity_r").shape == (3000,)
    sp.ReferenceSize = 100.0
    sp.UpdatePipeline()
    o = sp.GetClientSideObject().GetOutputDataObject(0)
    pd = o.GetPartitionedDataSet(names.index("Stars")).GetPartition(0)
    big = ns.vtk_to_numpy(pd.GetPointData().GetArray("StarSize"))
    _, extent = mo.auto_extent(ns.vtk_to_numpy(pd.GetPoints().GetData()))
    np.testing.assert_allclose(big, size * 100.0 / extent, rtol=1e-6)  # proportional to ReferenceSize
    assert big[young].min() > big[~young].max()  # brighter -> bigger
    up_stars = galaxy.GetClientSideObject().GetOutputDataObject(0).GetPartitionedDataSet(1).GetPartition(0)
    assert up_stars.GetPointData().GetArray("StarRGB") is None  # input untouched


def test_follow_camera_links_all_space_filters(galaxy):
    view = pvs.CreateRenderView()
    view.ViewSize = [400, 400]
    s = pvs.GIZMOStarlightImage(Input=galaxy, Resolution=64)
    sp = pvs.GIZMOStarProperties(Input=galaxy)
    for f in (s, sp):
        live.follow_camera(f, view)
    cam = view.GetActiveCamera()
    cam.SetFocalPoint(1, 0, 0)
    cam.SetPosition(1, 0, 30)
    cam.SetViewUp(0, 1, 0)
    _, _, _, size = live.view_window(view)
    assert list(s.Normal) == pytest.approx([0, 0, 1]) and s.Size == pytest.approx(size)
    assert sp.ReferenceSize == pytest.approx(size)
    cam.SetPosition(1, 0, 15)  # zoom in: stars keep their on-screen size
    assert sp.ReferenceSize == pytest.approx(size / 2)
    pvs.Delete(view)


def test_presets_import(tmp_path):
    """Generated presets load into ParaView (isolated settings: ParaView persists imported presets)."""
    code = ("from pvgizmo import looks; from paraview import servermanager as sm; import paraview.simple as pvs\n"
            "added = looks.register_presets(); again = looks.register_presets()\n"
            "p = sm.vtkSMTransferFunctionPresets.GetInstance()\n"
            "assert all(p.HasPreset(looks.preset_name(n)) for n in looks.GENERATED), added\n"
            "lut = pvs.GetColorTransferFunction('x'); assert lut.ApplyPreset('hot (matplotlib, GIZMO)', True)\n"
            "print('PRESETS_OK', len(added), len(again))\n")
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path), HOME=str(tmp_path))
    out = subprocess.run([sys.executable if "pvpython" in sys.executable else "pvpython", "-c", code],
                         env=env, capture_output=True, text=True, timeout=120)
    assert f"PRESETS_OK {len(looks.GENERATED)} 0" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
