"""pvgizmo.stars and pvgizmo.looks: starlight, ray-traced extinction, image scaling, composites."""

import numpy as np
import pytest

from pvgizmo import looks, stars

FIRE_META = {"UnitMass_In_CGS": 1.989e43, "UnitLength_In_CGS": 3.085678e21, "UnitVelocity_In_CGS": 1e5,
             "HubbleParam": 0.7, "Omega_Matter": 0.3, "Omega_Lambda": 0.7, "Time": 1.0, "Cosmological": 1,
             "PhysicalUnits": 1}


# --------------------------------------------------------------------------- stellar populations
def test_ssp_lookup_matches_pfh_interpolation():
    """pfh_python's colors_table interpolates linearly in (age index, Z index) with
    scipy.ndimage.map_coordinates(order=1); within the table that is what we do."""
    from scipy.ndimage import map_coordinates
    t = stars.ssp_table()
    rng = np.random.default_rng(0)
    age = 10 ** rng.uniform(-3, 1.1, 500)  # Gyr
    z = 10 ** rng.uniform(np.log10(2e-4), np.log10(0.04), 500)
    for band in ("u", "g", "r", "V", "K"):
        k = t["bands"].index(band)
        ia = np.interp(np.log10(age) + 9, t["log_age"], np.arange(len(t["log_age"])))
        iz = np.interp(np.log10(z), t["logz"], np.arange(len(t["logz"])))
        ref = 10 ** map_coordinates(t["log_lm"][k].astype(np.float64), (ia, iz), order=1)
        np.testing.assert_allclose(stars.ssp_light_to_mass(band, age, z), ref, rtol=1e-6)


def test_ssp_fades_and_reddens_with_age():
    ages = np.array([0.01, 0.1, 1.0, 10.0])
    lr = stars.ssp_light_to_mass("r", ages, 0.02)
    lu = stars.ssp_light_to_mass("u", ages, 0.02)
    assert np.all(np.diff(lr) < 0)  # populations fade
    assert np.all(np.diff(lu / lr) < 0)  # and redden
    # solar-type light is white by construction (L in L_sun of each band): a ~5 Gyr solar
    # population is within a factor of a few of equal u, g, r
    g = stars.ssp_light_to_mass("g", 5.0, 0.02)
    assert 0.2 < stars.ssp_light_to_mass("u", 5.0, 0.02) / g < 2 and 0.5 < stars.ssp_light_to_mass("r", 5.0, 0.02) / g < 3


def test_cosmic_time_and_ages():
    assert stars.cosmic_time_gyr(1.0, 0.7, 0.3) == pytest.approx(13.47, abs=0.03)  # flat LCDM, H0=70
    age = stars.stellar_age_gyr(np.array([1.0, 0.5, 0.1]), FIRE_META)
    assert age[0] == pytest.approx(0.0, abs=1e-9)
    assert age[1] == pytest.approx(stars.cosmic_time_gyr(1.0, 0.7, 0.3) - stars.cosmic_time_gyr(0.5, 0.7, 0.3), rel=1e-9)
    assert age[2] > 12.5
    # non-cosmological: StellarFormationTime in code time units (kpc/(km/s) = 0.978 Gyr)
    meta = dict(FIRE_META, Cosmological=0, HubbleParam=1.0, Time=2.0)
    np.testing.assert_allclose(stars.stellar_age_gyr([1.0, 2.0], meta), [0.9778, 0.0], rtol=1e-3)


def test_star_luminosities_populations_and_individual_stars():
    m = np.full(3, 1e-5)  # 1e5 Msun in FIRE units
    L, age = stars.star_luminosities(("u", "g", "r"), m, FIRE_META, sft=np.array([0.999, 0.5, 0.2]),
                                     z=np.full(3, 0.02))
    assert L.shape == (3, 3) and age is not None
    assert np.all(np.diff(L[:, 2]) < 0)  # older -> fainter
    assert L[0, 0] / L[0, 2] > L[2, 0] / L[2, 2]  # young -> bluer
    np.testing.assert_allclose(L[:, 2], 1e5 * np.clip(stars.ssp_light_to_mass("r", age, 0.02 + 2e-8), 0, 300))
    # individual stars: blackbody split of the given luminosity; the Sun is white
    ms = np.array([0.3, 1.0, 20.0])
    Ls = stars.zams_luminosity(ms)
    Lb, age = stars.star_luminosities(("u", "g", "r"), ms, {"UnitMass_In_CGS": 1.989e33}, lum_solar=Ls)
    assert age is None
    np.testing.assert_allclose(Lb[1], Ls[1], rtol=0.05)
    rgb = stars.star_rgb(Lb)
    assert rgb[0, 0] > rgb[0, 2] and rgb[2, 2] > rgb[2, 0]  # M dwarf red, O/B star blue
    assert np.all(rgb[1] > 250)


def test_zams_and_teff():
    assert stars.zams_luminosity(1.0) == pytest.approx(1.0)
    assert stars.teff(1.0, 1.0) == pytest.approx(stars.TSUN)
    t = stars.teff(stars.zams_luminosity(np.array([0.2, 1, 5, 40])), np.array([0.2, 1, 5, 40]))
    assert np.all(np.diff(t) > 0)


# --------------------------------------------------------------------------- ray tracing
def _naive_raytrace(x, h, mext, lum, size, res, kappa):
    """Literal re-implementation of pfh_python's raytrace_rgb (on our node grid, view along +z)."""
    ktab = stars._kernel_table()
    dx = size / (res - 1)
    grid = np.arange(res) * dx - size / 2
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    emis = np.zeros((res, res, lum.shape[1]))
    hmin, hmax = 0.5 * np.sqrt(2) * dx, 100 * np.sqrt(2) * dx
    for i in np.argsort(x[:, 2], kind="stable"):
        hh = max(h[i], hmin)
        hs = min(1.4 * hh, hmax)
        r2 = (gx - x[i, 0]) ** 2 + (gy - x[i, 1]) ** 2
        q = r2 / hh**2 * (1000 / 1.4**2)
        k = np.minimum(q.astype(int), 999)
        wk = (ktab[k] + (ktab[k + 1] - ktab[k]) * (q - k)) / hh**2
        wk[(r2 >= hs**2) | (q.astype(int) >= 1000)] = 0.0
        if mext[i] > 0:
            emis *= np.exp(-kappa[None, None, :] * (mext[i] * wk)[..., None])
        emis += lum[i][None, None, :] * wk[..., None]
    return emis


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    ng, ns_ = 400, 150
    xg = rng.normal(size=(ng, 3)) * [1.0, 1.0, 0.3]
    xs = rng.normal(size=(ns_, 3)) * [0.7, 0.7, 0.2]
    x = np.r_[xs, xg]
    h = np.r_[0.02 + 0.05 * rng.random(ns_), 0.1 + 0.3 * rng.random(ng)]
    mext = np.r_[np.zeros(ns_), 1e-4 * rng.random(ng)]
    lum = np.r_[rng.random((ns_, 3)), np.zeros((ng, 3))]
    return x, h, mext, lum, np.array([300.0, 200.0, 100.0])


def test_raytrace_matches_reference(scene):
    """Same sums as pfh_python's raytrace_rgb (separately verified against its compiled C code to
    float32 precision), threads included: depth slabs composited back to front are exact."""
    x, h, mext, lum, kappa = scene
    ref = _naive_raytrace(x, h, mext, lum, 4.0, 48, kappa)
    for nthreads in (1, 3):
        img, col = stars.raytrace(x, h, mext, lum, [0, 0, 0], np.eye(3), 4.0, 48, kappa, nthreads=nthreads)
        np.testing.assert_allclose(img, ref, rtol=1e-9, atol=1e-12 * ref.max())
    dimmed = 1 - img.sum(axis=(0, 1)) / stars.raytrace(x, h, 0 * mext, lum, [0, 0, 0], np.eye(3), 4.0, 48,
                                                       kappa)[0].sum(axis=(0, 1))
    assert 0.05 < dimmed[2] < dimmed[1] < dimmed[0] < 0.95  # partial, and bluer light absorbed more


def test_raytrace_screen_slab():
    """A star behind a uniform screen is dimmed by exactly exp(-kappa Sigma); one in front is not."""
    n = 60
    g = np.linspace(-3, 3, n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    screen = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])
    hs = np.full(n * n, 0.5)
    # sum of kernel weights of a lattice of spacing d is ~ 1/d^2: Sigma = m / d^2
    d = g[1] - g[0]
    m = np.full(n * n, 1e-3)
    star = np.array([[0.0, 0.0, -1.0], [0.3, 0.0, 1.0]])  # behind, in front (camera on +z)
    x = np.r_[star, screen]
    h = np.r_[np.full(2, 0.05), hs]
    mext = np.r_[np.zeros(2), m]
    kappa = np.array([500.0])
    sigma = 1e-3 / d**2
    behind = np.zeros((2 + n * n, 1))
    behind[0, 0] = 1.0
    img_b, col = stars.raytrace(x, h, mext, behind, [0, 0, 0], np.eye(3), 1.0, 41, kappa)
    img_0, _ = stars.raytrace(x, h, 0 * mext, behind, [0, 0, 0], np.eye(3), 1.0, 41, kappa)
    assert col[20, 20] == pytest.approx(sigma, rel=0.01)
    assert img_b.sum() / img_0.sum() == pytest.approx(np.exp(-kappa[0] * sigma), rel=0.03)
    front = np.zeros_like(behind)
    front[1, 0] = 1.0
    img_f, _ = stars.raytrace(x, h, mext, front, [0, 0, 0], np.eye(3), 1.0, 41, kappa)
    img_f0, _ = stars.raytrace(x, h, 0 * mext, front, [0, 0, 0], np.eye(3), 1.0, 41, kappa)
    np.testing.assert_allclose(img_f, img_f0)


def test_raytrace_rotation_and_depth(scene):
    x, h, mext, lum, kappa = scene
    rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # view along +x
    a, _ = stars.raytrace(x, h, mext, lum, [0, 0, 0], rot, 4.0, 32, kappa)
    b, _ = stars.raytrace(x[:, [2, 0, 1]], h, mext, lum, [0, 0, 0], np.eye(3), 4.0, 32, kappa)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-14)
    thin, _ = stars.raytrace(x, h, mext, lum, [0, 0, 0], np.eye(3), 4.0, 32, kappa, depth=0.2)
    keep = np.abs(x[:, 2]) < 0.1
    ref, _ = stars.raytrace(x[keep], h[keep], mext[keep], lum[keep], [0, 0, 0], np.eye(3), 4.0, 32, kappa)
    np.testing.assert_allclose(thin, ref)


# --------------------------------------------------------------------------- image scaling
def _pfh_nasa(maps, maxden, dynrange, f_saturated=1e-4):
    """make_threeband_image_process_bandmaps (color_scheme_nasa) from pfh_python, transcribed."""
    b, g, r = (maps[:, :, k].astype(np.float64).copy() for k in range(3))
    maxnorm, minnorm = maxden, maxden / dynrange
    i = (r + g + b) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(r / minnorm) / np.log(maxnorm / minnorm)
        g = np.log(g / minnorm) / np.log(maxnorm / minnorm)
        b = np.log(b / minnorm) / np.log(maxnorm / minnorm)
    bad = i <= 0
    r[bad] = g[bad] = b[bad] = 0.0
    x0 = int(np.round(f_saturated * (r.size - 1)))
    maxrgb = max(sorted(v.ravel(), reverse=True)[x0] for v in (r, g, b))
    if maxrgb > 1:
        r, g, b = r / maxrgb, g / maxrgb, b / maxrgb

    def clip(x):
        x = x * 253 + 2
        x[x >= 255] = 255
        x[x <= 2] = 2
        x[np.isnan(x)] = 2
        return x
    return np.stack([clip(r), clip(g), clip(b)], axis=-1)


def test_threeband_matches_pfh():
    rng = np.random.default_rng(1)
    maps = 10 ** rng.uniform(-3, 3, (40, 40, 3))
    maps[:5] = 0.0
    maxden, minden = stars.threeband_levels(maps, saturation=1e-2, dynamic_range=1e4)
    ref = _pfh_nasa(maps, maxden, 1e4)
    np.testing.assert_allclose(stars.threeband_rgb(maps, maxden, minden).astype(float), np.round(ref), atol=1)
    s = np.sort(maps.sum(axis=2).ravel())[::-1]
    s = s[s > 0]
    assert maxden == s[int(round(1e-2 * s.size))]
    sdss = stars.threeband_rgb(maps, maxden, minden, scheme="sdss")
    assert sdss.dtype == np.uint8 and sdss.shape == (40, 40, 3)


# --------------------------------------------------------------------------- composites, presets
def test_hsv_composite_matches_vizmo_shader():
    """vizmo composite.wgsl: rgb = cmap(t2); hsv.v *= t1."""
    import colorsys
    rng = np.random.default_rng(2)
    light = 10 ** rng.uniform(-2, 2, (16, 16))
    hue = 10 ** rng.uniform(1, 6, (16, 16))
    light[0, 0] = 0.0  # empty pixel -> black
    rgb, lr, hr = looks.hsv_composite(light, hue, "magma", (0.1, 50.0), (100.0, 1e5), True, True)
    table = looks.colormap_table("magma")
    for i, j in [(1, 1), (5, 9), (15, 3)]:
        t1 = np.clip((np.log10(light[i, j]) - np.log10(0.1)) / (np.log10(50.0) - np.log10(0.1)), 0, 1)
        t2 = np.clip((np.log10(hue[i, j]) - 2) / 3, 0, 1)
        h, s, v = colorsys.rgb_to_hsv(*table[int(round(t2 * 255))])
        ref = 255 * np.array(colorsys.hsv_to_rgb(h, s, v * t1))
        np.testing.assert_allclose(rgb[i, j], ref, atol=1)
    assert np.all(rgb[0, 0] == 0) and lr == (0.1, 50.0)
    auto, lr, hr = looks.hsv_composite(light, hue)
    assert lr[0] > 0 and hr[1] > hr[0] and auto.dtype == np.uint8


def test_preset_json():
    p = looks.preset_json("hot")
    assert p["Name"] == "hot (matplotlib, GIZMO)" and len(p["RGBPoints"]) == 64 * 4
    assert p["RGBPoints"][:4] == pytest.approx([0.0, 0.0416, 0.0, 0.0], abs=1e-3)
    assert looks.preset_name("magma") == "Magma (matplotlib)"


def test_raytrace_counting_sort_path(scene, monkeypatch):
    """Large inputs are ordered by an O(N) counting sort over depth bins: same image."""
    x, h, mext, lum, kappa = scene
    exact, _ = stars.raytrace(x, h, mext, lum, [0, 0, 0], np.eye(3), 4.0, 40, kappa)
    monkeypatch.setattr(stars, "EXACT_SORT_MAX", 0)
    binned, _ = stars.raytrace(x.astype(np.float32), h, mext, lum, [0, 0, 0], np.eye(3), 4.0, 40, kappa)
    np.testing.assert_allclose(binned, exact, rtol=1e-5, atol=1e-6 * exact.max())
