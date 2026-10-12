"""Phase 4 benchmark: where does the time go, as a function of N, resolution and zoom.

    nix develop -c pvpython bench/bench.py [N ...]        (default: 1e6 4e6 16e6)

Writes testdata/bench_<N>.hdf5 once (git-ignored), then times the reader, the
projection pipeline (with a breakdown of pvgizmo.meshoid_ops.project), slices,
3D grids and derived fields, through the actual ParaView filters. Prints a
markdown table; results go in docs/FINDINGS.md.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import pvgizmo  # noqa: E402,F401
import snapgen  # noqa: E402
from pvgizmo import io  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

ROWS = []


def rerun(proxy):
    """Force a filter/reader to re-execute (MarkModified alone keeps an up-to-date pipeline)."""
    proxy.GetClientSideObject().Modified()
    proxy.UpdatePipeline()


def timed(label, fn, n_rep=1):
    fn()  # warm-up (numba JIT / caches)
    t = time.perf_counter()
    for _ in range(n_rep):
        out = fn()
    dt = (time.perf_counter() - t) / n_rep
    ROWS.append((NLABEL, label, dt))
    print(f"  {label:<58s} {dt:8.3f} s", flush=True)
    return out


def breakdown(pos, h, m, size, res):
    """Split one projection call into its stages (same code path as mo.project)."""
    from meshoid.grid_deposition import GridSurfaceDensityMulti
    t0 = time.perf_counter()
    B = mo.view_basis((1, 2, 3), (0, 0, 1))
    x = mo.to_view(pos, np.zeros(3), B)
    t1 = time.perf_counter()
    hc = np.clip(np.asarray(h, dtype=np.float64), 2.0 * size / res, None)
    f = np.asarray(m, dtype=np.float64).reshape(-1, 1)
    t2 = time.perf_counter()
    GridSurfaceDensityMulti(f, x, hc, np.zeros(3), size, res, box_size=-1)
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2


def main(sizes):
    global NLABEL
    import paraview.simple as pvs
    os.makedirs(os.path.join(ROOT, "testdata"), exist_ok=True)
    for n in sizes:
        NLABEL = f"{n / 1e6:g}M"
        path = os.path.join(ROOT, "testdata", f"bench_{n}")
        if not os.path.exists(path + ".hdf5"):
            t = time.perf_counter()
            snapgen.make_big(path, n)
            print(f"wrote {path}.hdf5 in {time.perf_counter() - t:.1f} s")
        print(f"N = {NLABEL}")
        f = path + ".hdf5"
        parts = io.snapshot_parts(f)
        d = timed("io.read_ptype (Masses, SmoothingLength, Density)",
                  lambda: io.read_ptype(parts, 0, ["Masses", "SmoothingLength", "Density"]))
        pos, h, m = d["Coordinates"], d["SmoothingLength"], d["Masses"]
        r = pvs.GIZMOSnapshotReader(FileNames=[f])
        timed("reader: re-execute (default arrays, local disk)", lambda: rerun(r))
        for zoom in (1, 10, 100):
            size = 6.0 / zoom
            for res in (128, 512):
                timed(f"project() size={size:g} res={res}",
                      lambda: mo.project("SurfaceDensity", pos, h, np.zeros(3), size, res,
                                         normal=(1, 2, 3), view_up=(0, 0, 1), field=m))
            a, b, c = breakdown(pos, h, m, size, 512)
            print(f"    breakdown res=512: rotate {a:.3f}  clip/cast {b:.3f}  deposit {c:.3f}")
        p = pvs.MeshoidProjection(Input=r, AutoWindow=0, Center=[0, 0, 0], Size=6.0, Resolution=256)
        state = {"k": 0}

        def reaim():  # what a camera move does: change the view, re-execute
            state["k"] += 1
            p.Normal = [1.0, 0.1 * state["k"], 0.3]
            p.UpdatePipeline()
        timed("MeshoidProjection filter re-aim (256^2, full window)", reaim, n_rep=3)
        p.Resolution = 128
        timed("MeshoidProjection filter re-aim (128^2 = interactive)", reaim, n_rep=3)
        s = pvs.MeshoidSlice(Input=r, AutoWindow=0, Center=[0, 0, 0], Size=6.0, Resolution=256)
        t = time.perf_counter()
        s.UpdatePipeline()
        ROWS.append((NLABEL, "MeshoidSlice first (tree build + 256^2)", time.perf_counter() - t))
        print(f"  {'MeshoidSlice first (tree build + 256^2)':<58s} {ROWS[-1][2]:8.3f} s")

        def reslice():
            state["k"] += 1
            s.Normal = [1.0, 0.1 * state["k"], 0.3]
            s.UpdatePipeline()
        timed("MeshoidSlice re-aim (cached tree, 256^2)", reslice, n_rep=3)
        g = pvs.MeshoidGrid(Input=r, AutoWindow=0, Center=[0, 0, 0], Size=6.0, Resolution=128)
        timed("MeshoidGrid deposit 128^3", lambda: rerun(g))
        if n <= 4_000_000:
            dv = pvs.MeshoidDerivedFields(Input=r)
            t = time.perf_counter()
            dv.UpdatePipeline()
            ROWS.append((NLABEL, "MeshoidDerivedFields grad(Density) (tree cached)", time.perf_counter() - t))
            print(f"  {'MeshoidDerivedFields grad(Density) (tree cached)':<58s} {ROWS[-1][2]:8.3f} s")
        for x in (dv if n <= 4_000_000 else None, g, s, p, r):
            if x is not None:
                pvs.Delete(x)
    print("\n| N | stage | time [s] |\n|---|---|---|")
    for n, label, dt in ROWS:
        print(f"| {n} | {label} | {dt:.3f} |")


if __name__ == "__main__":
    args = [int(float(a)) for a in sys.argv[1:]] or [1_000_000, 4_000_000, 16_000_000]
    main(args)
