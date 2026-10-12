"""3D kernel deposition that is correct for kernels crossing the grid boundary.

meshoid.grid_deposition.GridDensity normalises each particle's kernel weights over
the nodes *inside* the grid and allocates a (2H/dx)^3 cache per particle. On
zoom/cutout data both break: a window around the zoom region overlaps outer
cells whose kernels are 10^4-10^5 voxels across, each of which then dumps its
whole mass into the box (m12f snapshot 177: 440x the mass actually inside) and
needs a petabyte-scale cache. Here each particle is normalised over its *full*
kernel instead -- the exact discrete sum over the unclipped node lattice for
small kernels (so fully-inside particles reproduce meshoid exactly), the
analytic integral (kernel3d is unit-normalised) for kernels > ~15 voxels -- and
only the in-grid nodes are visited. No per-particle allocation.
"""

import os
from threading import Thread

import numpy as np
from numba import njit

import pvgizmo  # noqa: F401  (multiprocessing executable before numba)
from meshoid.kernel_density import kernel3d

DISCRETE_NORM_MAX = 15.0  # kernel radius (in voxels) up to which the norm is summed exactly


@njit(fastmath=True, cache=True, nogil=True)
def _deposit3d_core(f, x, h, origin, dx, res, discrete_max):
    grid = np.zeros((res, res, res))
    for i in range(len(f)):
        hs = h[i]
        r = hs / dx  # kernel radius in voxels
        u0 = (x[i, 0] - origin[0]) / dx
        u1 = (x[i, 1] - origin[1]) / dx
        u2 = (x[i, 2] - origin[2]) / dx
        lo0, hi0 = int(np.floor(u0 - r)) + 1, int(np.floor(u0 + r))
        lo1, hi1 = int(np.floor(u1 - r)) + 1, int(np.floor(u1 + r))
        lo2, hi2 = int(np.floor(u2 - r)) + 1, int(np.floor(u2 + r))
        c0, d0 = max(lo0, 0), min(hi0, res - 1)
        c1, d1 = max(lo1, 0), min(hi1, res - 1)
        c2, d2 = max(lo2, 0), min(hi2, res - 1)
        if c0 > d0 or c1 > d1 or c2 > d2:
            continue  # kernel misses the grid
        rinv = 1.0 / r
        if r <= discrete_max:  # exact discrete normalisation over the full kernel
            tot = 0.0
            for a in range(lo0, hi0 + 1):
                da = (u0 - a) * (u0 - a)
                for b in range(lo1, hi1 + 1):
                    db = (u1 - b) * (u1 - b)
                    for c in range(lo2, hi2 + 1):
                        dc = (u2 - c) * (u2 - c)
                        tot += kernel3d(np.sqrt(da + db + dc) * rinv)
        else:  # well resolved: sum ~ integral = r^3 (kernel3d integrates to 1 over the unit ball)
            tot = r * r * r
        if tot <= 0.0:
            continue
        fac = f[i] / tot
        for a in range(c0, d0 + 1):
            da = (u0 - a) * (u0 - a)
            for b in range(c1, d1 + 1):
                db = (u1 - b) * (u1 - b)
                for c in range(c2, d2 + 1):
                    dc = (u2 - c) * (u2 - c)
                    w = kernel3d(np.sqrt(da + db + dc) * rinv)
                    if w > 0.0:
                        grid[a, b, c] += w * fac
    return grid / (dx * dx * dx)


def deposit3d(f, x, h, origin, dx, res, nthreads=None):
    """Density of conserved quantity f on the node grid origin + (i, j, k) dx, i,j,k < res.

    f (N,), x (N, 3), h (N,) kernel support radii (no clipping applied here).
    Threaded over particle chunks (the numba core releases the GIL).
    """
    f = np.ascontiguousarray(f, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    # skip particles whose kernel cannot reach the grid
    hi = origin + (res - 1) * dx
    reach = np.all((x > origin - h[:, None]) & (x < hi + h[:, None]), axis=1)
    f, x, h = f[reach], x[reach], h[reach]
    if len(f) == 0:
        return np.zeros((res, res, res))
    nthreads = nthreads or max(1, min(8, os.cpu_count() or 1, len(f) // 1000 + 1))
    # interleave so huge kernels (cost ~ res^3 each) spread over threads
    chunks = [slice(k, None, nthreads) for k in range(nthreads)]
    results = [None] * nthreads
    errors = []

    def work(k):
        try:
            s = chunks[k]
            results[k] = _deposit3d_core(f[s], x[s], h[s], origin, float(dx), int(res), DISCRETE_NORM_MAX)
        except Exception as e:  # surface worker errors instead of a None result
            errors.append(e)

    threads = [Thread(target=work, args=(k,)) for k in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return np.sum(results, axis=0)
