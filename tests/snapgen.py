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


def write_snapshot(path_base, parts, mtab, time=0.0, redshift=0.0, hubble=1.0,
                   cosmological=False, nfiles=1, boxsize=0.0):
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
            for u, v in UNITS.items():
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
    return outdir


if __name__ == "__main__":
    print(make_standard_set(sys.argv[1] if len(sys.argv) > 1 else "testdata"))
