"""Starlight: per-band stellar luminosities and dust-attenuated three-band images.

Follows PFH's FIRE visualisation pipeline (pfh_python: colors_sps/colors_table.py,
visualization/raytrace_projection.py + c_libraries/RayTrace_RGB,
visualization/make_threeband_image.py), so images look like the familiar FIRE
"mock Hubble" pictures:

* Star particles are stellar populations. L/M in each band comes from the
  BC03 Chabrier-IMF SSP table shipped with pfh_python (pvgizmo/data/ssp_chabrier.npz,
  converted from colors.chabrier.hdf5), as a function of age and metallicity.
  Units are L_sun *in that band*, so solar-type light comes out white.
* Individual stars (STARFORGE sinks) use StarLuminosity_Solar when present, or
  the ZAMS L(M) fit used by vizmo otherwise, with the same L in every band.
* Gas metals absorb along the line of sight with per-band opacities from PFH's
  attenuation/cross_section.py (Pei 1992 SMC-like dust, per solar metallicity).
* The image is a depth-ordered emission/absorption sweep (raytrace_rgb), then
  log-scaled to RGB as in make_threeband_image ("NASA" colours).
"""

import os
from threading import Thread

import numpy as np
from numba import njit

import pvgizmo  # noqa: F401  (multiprocessing executable before numba)

MSUN_G = 1.989e33
KPC_CM = 3.085678e21
GYR_S = 3.15576e16

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ssp_chabrier.npz")
_SSP = {}

# effective wavelength (Angstrom) and dust opacity per solar metallicity (cm^2/g of gas),
# evaluated from pfh_python attenuation.cross_section.opacity_per_solar_metallicity
BANDS = {
    "Bol": (4243.93, 74.138), "U": (3600.0, 89.105), "B": (4400.0, 71.048), "V": (5556.0, 52.745),
    "R": (6940.0, 38.227), "I": (8700.0, 26.642), "J": (12150.0, 14.905), "H": (16540.0, 8.450),
    "K": (21790.0, 5.043), "u": (3551.0, 90.425), "g": (4686.0, 65.821), "r": (6165.0, 45.577),
    "i": (7481.0, 34.025), "z": (8931.0, 25.501),
}
# PFH stellar_raytrace scales the three opacities (blue -> red channel order) by these
KAPPA_TWEAK = (1.0, 0.75, 0.5)
ZSUN = 0.02
BAND_SETS = (("u", "g", "r"), ("B", "V", "I"), ("U", "V", "K"), ("g", "r", "i"))


def ssp_table():
    if not _SSP:
        d = np.load(_DATA)
        _SSP.update(log_age=d["log_age_yr"], logz=np.log10(d["metallicity"]), log_lm=d["log_lm"],
                    bands=[str(b) for b in d["bands"]])
    return _SSP


def ssp_light_to_mass(band, age_gyr, z_abs):
    """L/M of a simple stellar population, in L_sun(band) per M_sun.

    Bilinear in (log age, log Z); ages and metallicities outside the table are clamped
    (PFH's colors_table uses the nearest node in stellar_raytrace, and extrapolates Z).
    """
    t = ssp_table()
    k = t["bands"].index(band)
    la = np.log10(np.maximum(np.asarray(age_gyr, np.float64), 1e-12)) + 9.0
    lz = np.log10(np.maximum(np.asarray(z_abs, np.float64), 1e-12))
    ia = np.interp(la, t["log_age"], np.arange(len(t["log_age"])))
    iz = np.interp(lz, t["logz"], np.arange(len(t["logz"])))
    tab = t["log_lm"][k]
    a0 = np.minimum(np.floor(ia).astype(int), tab.shape[0] - 2)
    z0 = np.minimum(np.floor(iz).astype(int), tab.shape[1] - 2)
    fa, fz = ia - a0, iz - z0
    v = ((1 - fa) * (1 - fz) * tab[a0, z0] + fa * (1 - fz) * tab[a0 + 1, z0]
         + (1 - fa) * fz * tab[a0, z0 + 1] + fa * fz * tab[a0 + 1, z0 + 1])
    return 10.0 ** v


def zams_luminosity(m_msun):
    """Piecewise main-sequence L(M) in L_sun (vizmo's fallback for sinks without a luminosity)."""
    m = np.asarray(m_msun, dtype=np.float64)
    return np.select([m < 0.43, m < 2.0, m < 55.0], [0.23 * m**2.3, m**4.0, 1.4 * m**3.5], 32000.0 * m)


TSUN = 5772.0


def teff(lum_solar, m_msun):
    """Effective temperature (K) from L and a ZAMS radius R = M^0.8 R_sun; clipped to 2500-60000 K."""
    r = np.maximum(np.asarray(m_msun, np.float64), 0.08) ** 0.8
    with np.errstate(invalid="ignore"):
        t = TSUN * (np.maximum(np.asarray(lum_solar, np.float64), 0.0) / r**2) ** 0.25
    return np.clip(np.nan_to_num(t, nan=2500.0), 2500.0, 60000.0)


def blackbody_band_factor(t, band):
    """L_band / L_bol of a blackbody at t, relative to the Sun's: L in units of L_sun(band) is
    L_bol x this, so a solar-temperature star is white in any band set (like the SSP table)."""
    if band == "Bol":
        return np.ones_like(np.asarray(t, np.float64))
    x0 = 1.438777 / (BANDS[band][0] * 1e-8)  # hc / (lambda k) in K

    def f(tt):  # lambda B_lambda / (sigma T^4), up to constants
        with np.errstate(over="ignore"):
            return 1.0 / (np.expm1(x0 / tt) * tt**4)
    return f(np.asarray(t, np.float64)) / f(TSUN)


def cosmic_time_gyr(a, h, omega_m, omega_l=None):
    """Age of a flat LCDM universe at scale factor a (radiation neglected)."""
    omega_l = 1.0 - omega_m if omega_l is None else omega_l
    h0 = 100.0 * h * 1e5 / (1e3 * KPC_CM)  # s^-1
    a = np.asarray(a, dtype=np.float64)
    return 2.0 / (3.0 * h0 * np.sqrt(omega_l)) * np.arcsinh(np.sqrt(omega_l / omega_m) * a**1.5) / GYR_S


def header_value(meta, *names, default=None):
    for n in names:
        if n in meta and meta[n] is not None:
            v = np.ravel(meta[n])
            if v.size:
                return float(v[0])
    return default


def stellar_age_gyr(sft, meta):
    """Ages from StellarFormationTime and the snapshot header (`meta`: reader field data).

    Cosmological: sft is a scale factor; ages from flat LCDM. Otherwise sft is a time in
    code units (the reader's physical units already divided by h): age = Time/h - sft.
    """
    sft = np.asarray(sft, dtype=np.float64)
    h = header_value(meta, "HubbleParam", default=1.0)
    if header_value(meta, "Cosmological", default=0.0):
        om = header_value(meta, "Omega_Matter", "Omega0", default=0.3)
        ol = header_value(meta, "Omega_Lambda", "OmegaLambda", default=1.0 - om)
        a_now = header_value(meta, "Time")
        return np.maximum(cosmic_time_gyr(a_now, h, om, ol) - cosmic_time_gyr(np.clip(sft, 1e-8, None), h, om, ol),
                          0.0)
    t_unit = (header_value(meta, "UnitLength_In_CGS", default=KPC_CM)
              / header_value(meta, "UnitVelocity_In_CGS", default=1e5) / GYR_S)
    t_now = header_value(meta, "Time", default=0.0) / (h if header_value(meta, "PhysicalUnits", default=1) else 1.0)
    return np.maximum(t_now - sft, 0.0) * t_unit


def mass_unit_msun(meta):
    return header_value(meta, "UnitMass_In_CGS", default=1e10 * MSUN_G) / MSUN_G


def kappa_code(band, meta):
    """Opacity per solar metallicity in code units: tau = kappa * Sigma_metals/Zsun (code mass/length^2)."""
    um = header_value(meta, "UnitMass_In_CGS", default=1e10 * MSUN_G)
    ul = header_value(meta, "UnitLength_In_CGS", default=KPC_CM)
    return BANDS[band][1] * um / ul**2


def metallicity(arr):
    """Total metal mass fraction from a GIZMO Metallicity array (N,) or (N, k)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    return a[:, 0] if a.ndim == 2 else a


def star_luminosities(bands, masses, meta, sft=None, z=None, lum_solar=None, add_base_age=3e-4,
                      add_base_z=2e-8, max_lm=300.0):
    """(N, len(bands)) luminosities in L_sun(band) and the ages used (Gyr, or None).

    Stellar populations (sft given): SSP L/M x mass, like pfh_python's stellar_raytrace
    (base age/metallicity offsets and the L/M <= 300 cap included). Individual stars
    (lum_solar given, or neither: ZAMS L(M)): split into bands as a blackbody at the
    effective temperature from L and M (blackbody_band_factor), so hot stars are blue.
    """
    m_msun = np.asarray(masses, dtype=np.float64) * mass_unit_msun(meta)
    if lum_solar is not None or sft is None:
        lum = np.asarray(lum_solar, np.float64) if lum_solar is not None else zams_luminosity(m_msun)
        t = teff(lum, m_msun)
        return np.column_stack([lum * blackbody_band_factor(t, b) for b in bands]), None
    age = stellar_age_gyr(sft, meta) + add_base_age
    zz = np.full(len(m_msun), ZSUN) if z is None else np.where(z > 0, z + add_base_z, add_base_z)
    out = np.empty((len(m_msun), len(bands)))
    for i, b in enumerate(bands):
        out[:, i] = m_msun * np.clip(ssp_light_to_mass(b, age, zz), 0.0, max_lm)
    return out, age


def star_rgb(lums, gamma=0.5):
    """Per-star display colour (uchar RGB) from three band luminosities (blue, green, red order).

    Each star's colour is its band ratio normalised to its brightest band; gamma < 1 softens
    the saturation like the log scaling of the images.
    """
    l = np.maximum(np.asarray(lums, dtype=np.float64), 0.0)[:, ::-1]  # -> R, G, B
    c = l / np.maximum(l.max(axis=1, keepdims=True), 1e-300)
    return np.round(255 * c**gamma).astype(np.uint8)


# --------------------------------------------------------------------------- ray-traced image
_KTAB_N = 1000
_HKERN = 1.4  # PFH: kernel table out to 1.4 h (Gaussian fit to the projected quintic spline)


def _kernel_table():
    r2 = np.arange(_KTAB_N + 1) * (_HKERN**2 / _KTAB_N)
    k = (135.0 / (14.0 * np.pi)) * np.exp(-(135.0 / 14.0) * r2)
    k[-1] = 0.0
    return k


@njit(fastmath=True, cache=True, nogil=True)
def _raytrace_core(order, x, h, mext, lum, center, basis, size, res, kappa, ktab, hmin, hmax):
    """PFH raytrace_rgb on the node grid: particles visited back to front (`order`), each
    first dims the pixels under its kernel by exp(-kappa_b * mext * W) then adds lum_b * W.
    Returns (emission (res, res, nb), sum of mext*W (res, res)).
    """
    nb = lum.shape[1]
    emis = np.zeros((res, res, nb))
    col = np.zeros((res, res))
    dx = size / (res - 1)
    half = size / 2
    ntab = ktab.shape[0] - 1
    tab_inv = ntab / (_HKERN * _HKERN)
    b00, b01, b02 = basis[0, 0], basis[0, 1], basis[0, 2]
    b10, b11, b12 = basis[1, 0], basis[1, 1], basis[1, 2]
    for n in range(order.shape[0]):
        i = order[n]
        r0 = x[i, 0] - center[0]
        r1 = x[i, 1] - center[1]
        r2 = x[i, 2] - center[2]
        xs0 = r0 * b00 + r1 * b01 + r2 * b02 + half
        xs1 = r0 * b10 + r1 * b11 + r2 * b12 + half
        hh = h[i]
        if hh < hmin:
            hh = hmin
        h2_i = 1.0 / (hh * hh)
        hs = hh * _HKERN
        if hs > hmax:
            hs = hmax
        hs2 = hs * hs
        gxmin = max(int(np.floor((xs0 - hs) / dx)) + 1, 0)
        gxmax = min(int(np.floor((xs0 + hs) / dx)), res - 1)
        gymin = max(int(np.floor((xs1 - hs) / dx)) + 1, 0)
        gymax = min(int(np.floor((xs1 + hs) / dx)), res - 1)
        m = mext[i]
        for gx in range(gxmin, gxmax + 1):
            ddx = xs0 - gx * dx
            ddx *= ddx
            for gy in range(gymin, gymax + 1):
                ddy = xs1 - gy * dx
                r2n = ddx + ddy * ddy
                if r2n >= hs2:
                    continue
                q = r2n * h2_i * tab_inv
                k = int(q)
                if k >= ntab:
                    continue
                wk = h2_i * (ktab[k] + (ktab[k + 1] - ktab[k]) * (q - k))
                if m > 0.0:
                    d = m * wk
                    col[gx, gy] += d
                    for b in range(nb):
                        y = kappa[b] * d
                        # most sightline elements are optically thin: skip the exp (error < y^3/6)
                        emis[gx, gy, b] *= (1.0 - y + 0.5 * y * y) if y < 1e-3 else np.exp(-y)
                for b in range(nb):
                    lb = lum[i, b]
                    if lb != 0.0:
                        emis[gx, gy, b] += lb * wk
    return emis, col


EXACT_SORT_MAX = 200_000  # above this, depth order comes from an O(N) counting sort
DEPTH_BINS = 1 << 20


@njit(cache=True, nogil=True)
def _cull_and_order(x, h, active, center, basis, half, hmin, hmax, depth, nbins, exact_max):
    """Indices of the active particles whose kernel reaches the window (and depth slab), back to
    front. Returns (order, depths): when more than exact_max survive, `order` is already sorted
    by a counting sort over nbins depth bins (ties within a bin keep index order); otherwise
    the caller sorts it by `depths` exactly.
    """
    n = x.shape[0]
    keep = np.empty(n, np.int64)
    w = np.empty(n)
    m = 0
    for i in range(n):
        if not active[i]:
            continue
        r0 = x[i, 0] - center[0]
        r1 = x[i, 1] - center[1]
        r2 = x[i, 2] - center[2]
        u = r0 * basis[0, 0] + r1 * basis[0, 1] + r2 * basis[0, 2]
        v = r0 * basis[1, 0] + r1 * basis[1, 1] + r2 * basis[1, 2]
        ww = r0 * basis[2, 0] + r1 * basis[2, 1] + r2 * basis[2, 2]
        hh = h[i] if h[i] > hmin else hmin
        reach = min(hh * _HKERN, hmax)
        if abs(u) >= half + reach or abs(v) >= half + reach:
            continue
        if depth > 0 and abs(ww) >= depth / 2:
            continue
        keep[m] = i
        w[m] = ww
        m += 1
    keep, w = keep[:m], w[:m]
    if m <= exact_max or m == 0:
        return keep, w
    wmin, wmax = w.min(), w.max()
    scale = (nbins - 1) / (wmax - wmin) if wmax > wmin else 0.0
    counts = np.zeros(nbins + 1, np.int64)
    bins = np.empty(m, np.int64)
    for k in range(m):
        b = int((w[k] - wmin) * scale)
        bins[k] = b
        counts[b + 1] += 1
    for b in range(nbins):
        counts[b + 1] += counts[b]
    out = np.empty(m, np.int64)
    for k in range(m):
        out[counts[bins[k]]] = keep[k]
        counts[bins[k]] += 1
    return out, w[:0]


def raytrace(x, h, mext, lum, center, basis, size, res, kappa, depth=0.0, hmin=None, hmax=None,
             nthreads=None, active=None):
    """Emission/absorption image seen along basis[2] (camera on the +basis[2] side).

    x (N, 3), h (N,) kernel lengths, mext (N,) extinguishing mass (gas metal mass / Zsun;
    0 for sources), lum (N, nb) source luminosities (0 for gas), kappa (nb,). `active` (N,) bool
    (default: mext > 0 or any lum != 0) can be passed in when cached.
    Returns (attenuated surface brightness (res, res, nb), extinguishing column (res, res)).
    Threads sweep contiguous depth ranges; their (emission, transmission) layers are then
    composited back to front, which is exact (absorption is multiplicative).
    """
    lum = np.asarray(lum, dtype=np.float64)
    if lum.ndim == 1:
        lum = lum[:, None]
    nb = lum.shape[1]
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    pix = size / (res - 1)
    hmin = 0.5 * np.sqrt(2.0) * pix if hmin is None else hmin  # PFH: half a pixel diagonal
    hmax = 100.0 * np.sqrt(2.0) * pix if hmax is None else hmax
    # cull to the window (plus kernel) and the depth slab, then order back (far) to front
    xa = np.ascontiguousarray(x)
    if xa.dtype not in (np.float32, np.float64):
        xa = xa.astype(np.float64)
    ha = np.ascontiguousarray(h, dtype=np.float64)
    ma = np.ascontiguousarray(mext, dtype=np.float64)
    if active is None:
        active = (ma > 0) | np.any(lum != 0, axis=1)
    order, w = _cull_and_order(xa, ha, active, center, basis, size / 2, float(hmin), float(hmax),
                               float(depth), DEPTH_BINS, EXACT_SORT_MAX)
    if order.size == 0:
        return np.zeros((res, res, nb)), np.zeros((res, res))
    if w.size:
        order = order[np.argsort(w, kind="stable")]
    ktab = _kernel_table()
    nthreads = nthreads or max(1, min(os.cpu_count() or 1, order.size // 50_000 + 1))
    bounds = np.linspace(0, order.size, nthreads + 1).astype(np.int64)
    results = [None] * nthreads
    errors = []

    def work(t):
        try:
            results[t] = _raytrace_core(order[bounds[t]:bounds[t + 1]], xa, ha, ma, lum, center, basis,
                                        float(size), int(res), kappa, ktab, float(hmin), float(hmax))
        except Exception as e:
            errors.append(e)

    threads = [Thread(target=work, args=(t,)) for t in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    emis, col = results[0]
    for e, c in results[1:]:  # each later layer is in front of the ones before it
        emis = emis * np.exp(-kappa[None, None, :] * c[:, :, None]) + e
        col = col + c
    return emis, col


# --------------------------------------------------------------------------- image scaling
def threeband_levels(maps, saturation=1e-4, dynamic_range=1e4):
    """(maxden, minden) for a (res, res, 3) band image: the brightness of the `saturation`
    fraction brightest pixel of r+g+b (PFH image_maker), and maxden / dynamic_range.
    """
    s = np.sort(np.ravel(maps.sum(axis=2)))
    s = s[(s > 0) & np.isfinite(s)]
    if s.size == 0:
        return 1.0, 1.0 / dynamic_range
    n0 = int(np.clip(np.round(saturation * s.size), 0, s.size - 1))
    maxden = s[::-1][n0]
    return maxden, maxden / dynamic_range


def threeband_rgb(maps, maxden, minden, scheme="nasa", f_saturated=1e-4):
    """uchar RGB from a (res, res, 3) image in (blue, green, red) band order.

    Port of make_threeband_image_process_bandmaps: 'nasa' = per-channel log between
    minden and maxden; 'sdss' = asinh stretch of the mean intensity, colours preserved.
    """
    b, g, r = (np.array(maps[:, :, k], dtype=np.float64) for k in range(3))
    i = (r + g + b) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if scheme == "sdss":
            q, alpha = 9.0, 0.3
            wt = np.arcsinh(alpha * q * (i / minden)) / q / i
            r, g, b = r * wt, g * wt, b * wt
        else:
            lr = np.log(maxden / minden)
            r, g, b = np.log(r / minden) / lr, np.log(g / minden) / lr, np.log(b / minden) / lr
    bad = ~(i > 0)
    for c in (r, g, b):
        c[bad | ~np.isfinite(c)] = 0.0
    if f_saturated > 0:  # rescale so that only this fraction of channel values exceeds 1
        x0 = int(round(f_saturated * (r.size - 1)))
        top = max(np.sort(np.ravel(c))[::-1][x0] for c in (r, g, b))
        if top > 1.0:
            r, g, b = r / top, g / top, b / top
    rgb = np.stack([r, g, b], axis=-1)
    return np.round(np.clip(rgb * 253.0 + 2.0, 2.0, 255.0)).astype(np.uint8)  # PFH clip_256: [2, 255]
