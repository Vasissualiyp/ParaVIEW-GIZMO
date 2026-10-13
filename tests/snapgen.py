"""Write small synthetic GIZMO-format HDF5 snapshots for tests and demos.

    python tests/snapgen.py OUTDIR      # writes the standard fixture set

Layout of the standard set:
    single/snapshot_000.hdf5            Plummer gas + stars + DM (MassTable masses)
    multi/snapdir_000/snapshot_000.{0,1,2}.hdf5   same content split in 3 parts
    series/snapshot_00{0,1,2}.hdf5      3 time steps (Time = 0, 0.1, 0.2)
    cosmo/snapshot_000.hdf5             cosmological header (a=0.5, h=0.7)
    zoom/snapshot_000.hdf5              tiny (1e-4) float64 cloud at x~1948, like the
                                        FIRE m12f cutouts: collapses in float32 unless recentred
"""

import os
import sys

import h5py
import numpy as np

UNITS = dict(UnitLength_In_CGS=3.085678e18, UnitMass_In_CGS=1.989e33, UnitVelocity_In_CGS=1e2)


ZOOM_CENTER = np.array([1948.06, 1916.5, 1993.33])


def plummer(n, a=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(1e-3, 0.99, n)
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    v = rng.normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1)[:, None]) * r[:, None]


def make_particles(ngas=4000, nstar=500, ndm=2000, seed=0, t=0.0):
    """Particle data for one snapshot: {ptype: {field: array}} plus a MassTable."""
    from meshoid import Meshoid

    rng = np.random.default_rng(seed)
    x = plummer(ngas, rng=rng) * (1.0 + t)  # expand a bit with time
    m = np.full(ngas, 1.0 / ngas)
    M = Meshoid(x, m, n_jobs=1)
    gas = dict(
        Coordinates=x,
        Velocities=rng.normal(size=(ngas, 3)),
        Masses=m,
        ParticleIDs=np.arange(1, ngas + 1, dtype=np.uint32),
        SmoothingLength=M.SmoothingLength(),
        Density=M.Density(),
        InternalEnergy=rng.uniform(1, 2, ngas),
        Metallicity=rng.uniform(0, 0.02, (ngas, 11)),
    )
    gas = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in gas.items()}
    stars = dict(
        Coordinates=plummer(nstar, 0.5, rng).astype(np.float32),
        Velocities=rng.normal(size=(nstar, 3)).astype(np.float32),
        Masses=np.full(nstar, 1e-3, np.float32),
        ParticleIDs=np.arange(ngas + 1, ngas + nstar + 1, dtype=np.uint32),
        StellarFormationTime=rng.uniform(0, max(t, 1e-3), nstar).astype(np.float32),
    )
    dm = dict(  # no Masses dataset: mass comes from the header MassTable
        Coordinates=plummer(ndm, 3.0, rng).astype(np.float32),
        Velocities=rng.normal(size=(ndm, 3)).astype(np.float32),
        ParticleIDs=np.arange(ngas + nstar + 1, ngas + nstar + ndm + 1, dtype=np.uint32),
    )
    mtab = np.zeros(6)
    mtab[1] = 5e-3
    return {0: gas, 1: dm, 4: stars}, mtab


FIRE_UNITS = dict(UnitLength_In_CGS=3.085678e21, UnitMass_In_CGS=1.989e43, UnitVelocity_In_CGS=1e5)


def make_galaxy(ngas=6000, nstar=3000, seed=5):
    """FIRE-like disc galaxy at z = 0 (kpc, 1e10 Msun, km/s; a = 1, h = 0.7).

    Stars: young (blue) ones in a thin star-forming ring, old (red) ones in a bulge;
    StellarFormationTime are scale factors. Gas: a dusty disc (solar metallicity)
    with a hole in the middle, so the bulge is seen unobscured face-on.
    """
    from meshoid import Meshoid

    rng = np.random.default_rng(seed)
    r = rng.uniform(2.0, 8.0, ngas)
    phi = rng.uniform(0, 2 * np.pi, ngas)
    xg = np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.normal(0, 0.15, ngas)])
    mg = np.full(ngas, 1.0 / ngas)  # 1e10 Msun: ~50 Msun/pc^2, tau_u ~ 1 face-on
    M = Meshoid(xg, mg, n_jobs=1)
    gas = dict(Coordinates=xg, Masses=mg, SmoothingLength=M.SmoothingLength(), Density=M.Density(),
               Velocities=np.zeros((ngas, 3)), InternalEnergy=rng.uniform(10, 1000, ngas),
               Temperature=10 ** rng.uniform(1, 6, ngas),
               Metallicity=np.column_stack([np.full(ngas, 0.02), np.zeros((ngas, 10))]))
    gas = {k: v.astype(np.float32) for k, v in gas.items()}
    nyoung = nstar // 2
    ry = rng.uniform(4.0, 6.0, nyoung)
    py = rng.uniform(0, 2 * np.pi, nyoung)
    young = np.column_stack([ry * np.cos(py), ry * np.sin(py), rng.normal(0, 0.1, nyoung)])
    old = plummer(nstar - nyoung, 1.0, rng)
    stars = dict(
        Coordinates=np.r_[young, old].astype(np.float32),
        Masses=np.full(nstar, 1e-5, np.float32),
        Velocities=np.zeros((nstar, 3), np.float32),
        # young: formed at a > 0.99 (< ~100 Myr ago); old: a < 0.3 (> ~10 Gyr ago)
        StellarFormationTime=np.r_[rng.uniform(0.993, 0.9999, nyoung),
                                   rng.uniform(0.15, 0.3, nstar - nyoung)].astype(np.float32),
        Metallicity=np.column_stack([np.r_[np.full(nyoung, 0.02), np.full(nstar - nyoung, 0.004)],
                                     np.zeros((nstar, 10))]).astype(np.float32),
    )
    return {0: gas, 4: stars}, np.zeros(6)


def make_sinks(ngas=3000, nsink=40, seed=6):
    """STARFORGE-like cloud with sink particles (PartType5) carrying StarLuminosity_Solar."""
    from meshoid import Meshoid

    rng = np.random.default_rng(seed)
    xg = plummer(ngas, 1.0, rng)
    mg = np.full(ngas, 1e3 / ngas)
    M = Meshoid(xg, mg, n_jobs=1)
    gas = dict(Coordinates=xg, Masses=mg, SmoothingLength=M.SmoothingLength(), Density=M.Density(),
               Velocities=np.zeros((ngas, 3)), InternalEnergy=rng.uniform(1, 2, ngas))
    gas = {k: v.astype(np.float32) for k, v in gas.items()}
    ms = 10 ** rng.uniform(-1, 1.5, nsink)
    sinks = dict(Coordinates=plummer(nsink, 0.5, rng).astype(np.float32), Masses=ms.astype(np.float32),
                 Velocities=np.zeros((nsink, 3), np.float32),
                 StarLuminosity_Solar=(ms**3.5).astype(np.float32))
    return {0: gas, 5: sinks}, np.zeros(6)


def write_snapshot(path_base, parts, mtab, time=0.0, redshift=0.0, hubble=1.0,
                   cosmological=False, nfiles=1, boxsize=0.0, units=None):
    """Write `parts` to `path_base`.hdf5 (nfiles == 1) or `path_base`.<k>.hdf5."""
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    ntot = np.zeros(6, dtype=np.int64)
    for t, d in parts.items():
        ntot[t] = len(d["Coordinates"])
    for k in range(nfiles):
        fname = f"{path_base}.hdf5" if nfiles == 1 else f"{path_base}.{k}.hdf5"
        nthis = np.zeros(6, dtype=np.int64)
        with h5py.File(fname, "w") as f:
            for t, d in parts.items():
                n = ntot[t]
                lo, hi = (k * n) // nfiles, ((k + 1) * n) // nfiles
                nthis[t] = hi - lo
                g = f.create_group(f"PartType{t}")
                for name, arr in d.items():
                    g.create_dataset(name, data=arr[lo:hi])
            h = f.create_group("Header").attrs
            h["NumPart_ThisFile"] = nthis.astype(np.int32)
            h["NumPart_Total"] = (ntot & 0xFFFFFFFF).astype(np.uint32)
            h["NumPart_Total_HighWord"] = (ntot >> 32).astype(np.uint32)
            h["MassTable"] = mtab
            h["Time"] = time
            h["Redshift"] = redshift
            h["BoxSize"] = boxsize
            h["NumFilesPerSnapshot"] = np.int32(nfiles)
            h["HubbleParam"] = hubble
            h["Omega_Matter"] = 0.3 if cosmological else 0.0
            h["Omega_Lambda"] = 0.7 if cosmological else 0.0
            h["ComovingIntegrationOn"] = np.int32(cosmological)
            for u, v in (units or UNITS).items():
                h[u] = v


def make_standard_set(outdir):
    parts, mtab = make_particles()
    write_snapshot(f"{outdir}/single/snapshot_000", parts, mtab)
    write_snapshot(f"{outdir}/multi/snapdir_000/snapshot_000", parts, mtab, nfiles=3)
    for i, t in enumerate((0.0, 0.1, 0.2)):
        p, mt = make_particles(seed=i, t=t)
        write_snapshot(f"{outdir}/series/snapshot_{i:03d}", p, mt, time=t)
    write_snapshot(f"{outdir}/cosmo/snapshot_000", parts, mtab, time=0.5, redshift=1.0,
                   hubble=0.7, cosmological=True, boxsize=10.0)
    zoom, zt = make_particles(seed=3)
    for t, d in zoom.items():
        scale = 1e-4 / 10  # Plummer extends to ~10 a
        d["Coordinates"] = ZOOM_CENTER + d["Coordinates"].astype(np.float64) * scale
        if "SmoothingLength" in d:
            d["SmoothingLength"] = d["SmoothingLength"] * np.float32(scale)
    write_snapshot(f"{outdir}/zoom/snapshot_000", zoom, zt)
    gal, gt = make_galaxy()
    write_snapshot(f"{outdir}/galaxy/snapshot_600", gal, gt, time=1.0, redshift=0.0, hubble=0.7,
                   cosmological=True, units=FIRE_UNITS)
    sk, st = make_sinks()
    write_snapshot(f"{outdir}/sinks/snapshot_010", sk, st)
    return outdir


if __name__ == "__main__":
    print(make_standard_set(sys.argv[1] if len(sys.argv) > 1 else "testdata"))


def make_big(path_base, n, background_frac=0.2, seed=0, des_ngb=32):
    """Large benchmark snapshot: Plummer sphere (a=1) + uniform background, gas only.

    Kernel radii and densities are analytic (no neighbour search, so 10^7+ particles
    take seconds), spanning ~4 decades like zoom/cutout data. float32 on disk.
    """
    rng = np.random.default_rng(seed)
    nb = int(n * background_frac)
    npl = n - nb
    m = np.full(n, 1.0 / n)
    x = np.empty((n, 3))
    x[:npl] = plummer(npl, rng=rng)
    L = 40.0
    x[npl:] = rng.uniform(-L / 2, L / 2, (nb, 3))
    r2 = np.sum(x**2, axis=1)
    rho = (1 - background_frac) * 3 / (4 * np.pi) * (1 + r2) ** -2.5 + background_frac / L**3
    H = (3 * des_ngb * m / (4 * np.pi * rho)) ** (1 / 3)
    gas = dict(Coordinates=x.astype(np.float32), Masses=m.astype(np.float32),
               SmoothingLength=H.astype(np.float32), Density=rho.astype(np.float32),
               Velocities=rng.normal(size=(n, 3)).astype(np.float32),
               InternalEnergy=rng.uniform(1, 2, n).astype(np.float32),
               ParticleIDs=np.arange(1, n + 1, dtype=np.uint32))
    write_snapshot(path_base, {0: gas}, np.zeros(6))
    return f"{path_base}.hdf5"
