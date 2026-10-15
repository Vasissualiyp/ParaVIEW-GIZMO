"""Meshoid-based projections and slices at arbitrary orientation (ParaView-independent).

Frame conventions
-----------------
A view is an orthonormal, right-handed basis (u, v, n): n is the line of sight
(projection axis / slice normal), v the image "up", u = v x n the image "right".
Images are (res, res[, k]) arrays indexed [i, j] at  center + s_i u + s_j v with
s = linspace(-size/2, size/2, res): a *node* grid spanning the window corner to
corner. That is the convention of meshoid's non-periodic deposition routines
(dx = size/(res-1), first index along x), which we call on coordinates rotated
into (u, v, n). (meshoid's own Slice/InterpToGrid use cell centres instead; our
slices use the node grid too, so slices and projections overlay exactly.)

Kernel radii are GIZMO's H (support radius). Projections need no neighbour
tree when H is known, so they call meshoid.grid_deposition directly; slices
reconstruct from the nearest particle and use a cached Meshoid (KD-tree).
"""

from collections import OrderedDict

import numpy as np

import pvgizmo  # noqa: F401  (multiprocessing executable before numba loads)
from meshoid import Meshoid

MODES = ("SurfaceDensity", "LineIntegral", "MassWeightedAverage", "ProjectedAverage")


def view_basis(normal, view_up=(0.0, 1.0, 0.0)):
    """(3,3) matrix with rows u, v, n for line of sight `normal` and up hint `view_up`."""
    n = np.asarray(normal, dtype=np.float64)
    if not np.linalg.norm(n) > 0:
        raise ValueError("normal must be non-zero")
    n = n / np.linalg.norm(n)
    up = np.asarray(view_up, dtype=np.float64)
    v = up - np.dot(up, n) * n
    if np.linalg.norm(v) < 1e-8 * max(np.linalg.norm(up), 1.0):  # up parallel to n: pick any
        trial = np.eye(3)[np.argmin(np.abs(n))]
        v = trial - np.dot(trial, n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    return np.array([u, v, n])


def grid_coords(size, res):
    """Node positions along one image axis (meshoid's non-periodic deposition grid)."""
    return np.linspace(-size / 2, size / 2, res)


def pixel_area(size, res):
    """Area per image node, so that sum(surface density) * pixel_area = deposited total."""
    return (size / (res - 1)) ** 2


def to_view(pos, center, basis):
    """Positions in the (u, v, n) frame, relative to `center`, in float64."""
    return (np.asarray(pos, dtype=np.float64) - np.asarray(center, dtype=np.float64)) @ basis.T


def auto_extent(pos):
    """meshoid's default window: median centre, size = 2 x 90th-percentile radius."""
    pos = np.asarray(pos, dtype=np.float64)
    c = np.median(pos, axis=0)
    r = np.sqrt(np.sum((pos - c) ** 2, axis=1))
    size = 2.0 * np.percentile(r, 90) if len(r) else 1.0
    return c, (size if size > 0 else 1.0)


def _channels(f):
    f = np.asarray(f, dtype=np.float64)
    return f.reshape(len(f), -1), f.shape[1:]


def projection_channels(mode, field, masses=None, volume=None):
    """View-independent per-particle channels of a projection; cache them across camera moves.

    Returns (chans (N, k), tail shape of the field). SurfaceDensity uses the field in place.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if field is None:
        raise ValueError(f"{mode} needs a field (e.g. Masses)")
    f = np.asarray(field)
    n, tail = len(f), f.shape[1:]
    f2 = f.reshape(n, -1)
    if mode == "SurfaceDensity":
        chans = f2
    elif mode == "LineIntegral":
        chans = f2 * np.asarray(volume, dtype=np.float64)[:, None]
    elif mode == "MassWeightedAverage":
        if masses is None:
            raise ValueError("MassWeightedAverage needs masses")
        mm = np.asarray(masses, dtype=np.float64)
        chans = np.column_stack([f2 * mm[:, None], mm])
    else:  # ProjectedAverage: number-weighted, like meshoid's GridAverage
        chans = np.column_stack([f2.astype(np.float64), np.ones(n)])
    return np.ascontiguousarray(chans), tail


def project_channels(mode, chans, tail, pos, h, center, size, res, normal=(0, 0, 1), view_up=(0, 1, 0),
                     smooth_fac=1.0, depth=0.0):
    """Deposit prepared channels along `normal` (the per-frame work of a camera move)."""
    from pvgizmo.deposit import project2d
    stacked = project2d(chans, pos, h, center, view_basis(normal, view_up), size, res,
                        smooth_fac=smooth_fac, depth=depth)
    if mode in ("MassWeightedAverage", "ProjectedAverage"):
        with np.errstate(invalid="ignore", divide="ignore"):
            stacked = stacked[..., :-1] / stacked[..., -1:]
    return stacked.reshape((res, res) + tuple(tail))


def project(mode, pos, h, center, size, res, normal=(0, 0, 1), view_up=(0, 1, 0), field=None,
            masses=None, volume=None, smooth_fac=1.0, depth=0.0, conservative=False, parallel=True):
    """Kernel-weighted projection along `normal`; returns a (res, res[, k]) array.

    mode:
      SurfaceDensity       integral of the density of conserved per-particle `field`
                           (e.g. Masses -> column density). Units: [field]/length^2.
      LineIntegral         integral of intensive `field` along the sightline
                           (= surface density of field * volume). Units: [field]*length.
      MassWeightedAverage  integral(rho f dl) / integral(rho dl), weights = `masses`.
      ProjectedAverage     particle-number-weighted average of `field` along the sightline.
    `field` may have several components (vectors are averaged per global component).
    h: GIZMO kernel radius H, clipped below at 2 pixels like meshoid does.
    depth > 0 keeps only particles with |distance along normal| < depth/2.
    Same deposit as meshoid's GridSurfaceDensity (pvgizmo.deposit.project2d rotates,
    culls and clips on the fly, so float32 inputs are never copied).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not (mode == "SurfaceDensity" and conservative):
        chans, tail = projection_channels(mode, field, masses, volume)
        return project_channels(mode, chans, tail, pos, h, center, size, res, normal, view_up,
                                smooth_fac, depth)
    # conservative: meshoid's conservative core renormalises a kernel over its in-window
    # pixels (dumping the whole mass of an edge-crossing particle inside) and allocates
    # (2H/dx)^2 per particle: use it only for kernels fully inside the window; the others
    # go through the non-renormalised routine (correct partial deposit).
    from meshoid.grid_deposition import GridSurfaceDensity
    if field is None:
        raise ValueError("SurfaceDensity needs a field (e.g. Masses)")
    basis = view_basis(normal, view_up)
    x = to_view(pos, center, basis)
    h = np.clip(smooth_fac * np.asarray(h, dtype=np.float64), 2.0 * size / res, None)
    if depth > 0:
        sel = np.abs(x[:, 2]) < depth / 2
        x, h = x[sel], h[sel]
    f, tail = _channels(field)
    if depth > 0:
        f = f[sel]
    origin = np.zeros(3)
    kw = dict(box_size=-1, parallel=parallel)
    inside = np.all(np.abs(x[:, :2]) + h[:, None] < size / 2, axis=1)
    out = np.zeros((res, res, f.shape[1]))
    for k in range(f.shape[1]):
        if inside.any():
            out[..., k] += GridSurfaceDensity(f[inside, k], x[inside], h[inside], origin, size, res,
                                              conservative=True, **kw)
        if (~inside).any():
            out[..., k] += GridSurfaceDensity(f[~inside, k], x[~inside], h[~inside], origin, size,
                                              res, conservative=False, **kw)
    return out.reshape((res, res) + tail)


class MeshoidCache:
    """Tiny LRU of Meshoid instances (KD-tree, neighbours, gradient weights)."""

    def __init__(self, maxsize=2):
        self.maxsize = maxsize
        self._d = OrderedDict()

    def get(self, key, pos, masses=None, kernel_radius=None):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        M = Meshoid(np.asarray(pos, dtype=np.float64),
                    None if masses is None else np.asarray(masses, dtype=np.float64),
                    kernel_radius=None if kernel_radius is None else np.asarray(kernel_radius, dtype=np.float64))
        self._d[key] = M
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return M

    def clear(self):
        self._d.clear()


def slice_points(center, size, res, normal=(0, 0, 1), view_up=(0, 1, 0)):
    """(res*res, 3) plane points, i (along u) slowest, matching image [i, j] order."""
    u, v, _ = view_basis(normal, view_up)
    s = grid_coords(size, res)
    si, sj = np.meshgrid(s, s, indexing="ij")
    return np.asarray(center, dtype=np.float64) + si.reshape(-1, 1) * u + sj.reshape(-1, 1) * v


def slice_field(meshoid, field, center, size, res, normal=(0, 0, 1), view_up=(0, 1, 0),
                order=0, slope_limiter=False):
    """Reconstruct `field` on a res x res plane; returns (res, res[, k])."""
    f = np.asarray(field, dtype=np.float64)
    pts = slice_points(center, size, res, normal, view_up)
    vals = meshoid.Reconstruct(f, pts, order=order, slope_limiter=slope_limiter)
    return vals.reshape((res, res) + f.shape[1:])


def image_geometry(center, size, res, normal=(0, 0, 1), view_up=(0, 1, 0)):
    """origin, spacing, direction (columns u, v, n) for a vtkImageData of the image."""
    basis = view_basis(normal, view_up)
    s0 = -size / 2
    origin = np.asarray(center, dtype=np.float64) + s0 * basis[0] + s0 * basis[1]
    dx = size / (res - 1)
    return origin, (dx, dx, dx), basis.T


# ----------------------------------------------------------------------------- 3D grids
def grid3d_points(center, size, res):
    """(res^3, 3) node positions, index (i, j, k) in C order (k fastest), axis-aligned."""
    s = grid_coords(size, res)
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    return np.asarray(center, dtype=np.float64) + np.c_[X.ravel(), Y.ravel(), Z.ravel()]


def grid3d_geometry(center, size, res):
    """origin, spacing of the axis-aligned node grid (same nodes as the 2D maps)."""
    dx = size / (res - 1)
    return np.asarray(center, dtype=np.float64) - size / 2, (dx, dx, dx)


def grid3d_deposit(pos, h, field, center, size, res, parallel=True):
    """Density of conserved per-particle `field` on a res^3 node grid; (res,res,res[,k]).

    Nodes coincide with the 2D projection/slice nodes (center - size/2 + k dx,
    dx = size/(res-1)). Kernels are clipped to >= 1 voxel like Meshoid.DepositToGrid.
    Uses pvgizmo.deposit, not meshoid's GridDensity: that one renormalises kernels
    that cross the box boundary onto the in-box nodes (dumping their whole mass
    inside) and allocates (2H/dx)^3 per particle -- both fatal on zoom data. For
    kernels inside the box the result is identical to meshoid's, and exactly
    mass-conserving (to ~1e-4 for kernels wider than 15 voxels).
    """
    from pvgizmo.deposit import deposit3d
    dx = size / (res - 1)
    x = np.asarray(pos, dtype=np.float64)
    h = np.clip(np.asarray(h, dtype=np.float64), dx, None)
    f, tail = _channels(field)
    origin = np.asarray(center, dtype=np.float64) - size / 2
    nthreads = None if parallel else 1
    out = np.stack([deposit3d(f[:, k], x, h, origin, dx, res, nthreads=nthreads)
                    for k in range(f.shape[1])], axis=-1)
    return out.reshape((res, res, res) + tail)


def safe_log10(values):
    """log10 with non-positive entries set one decade below the smallest positive value."""
    v = np.asarray(values, dtype=np.float64)
    pos = v[v > 0]
    floor = pos.min() / 10 if pos.size else 1e-300
    return np.log10(np.where(v > 0, v, floor))


def scaled_log10(values, decades=4.0, top_percentile=99.9, levels=None):
    """log10(values) mapped to [0, 1]: 1 at the top_percentile of the positive values,
    0 at `decades` below it (and for empty voxels). levels=(lo, hi): fixed log10 levels
    instead (0 at lo, 1 at hi). Float32."""
    v = np.asarray(values, dtype=np.float64)
    pos = v[v > 0]
    if levels is not None:
        top, decades = float(levels[1]), float(levels[1]) - float(levels[0])
    elif pos.size == 0:
        return np.zeros(v.shape, np.float32)
    else:
        top = np.log10(np.percentile(pos, top_percentile))
    with np.errstate(divide="ignore"):
        lg = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), -np.inf)
    return np.clip((lg - (top - decades)) / decades, 0.0, 1.0).astype(np.float32)


def grid3d_interp(meshoid, field, center, size, res, order=0, slope_limiter=False):
    """Intensive `field` reconstructed at the res^3 nodes; (res,res,res[,k])."""
    f = np.asarray(field, dtype=np.float64)
    vals = meshoid.Reconstruct(f, grid3d_points(center, size, res), order=order, slope_limiter=slope_limiter)
    return vals.reshape((res, res, res) + f.shape[1:])


# ----------------------------------------------------------------------------- derived fields
DERIVED = ("KernelRadius", "Density", "Volume", "Gradient", "DivCurl", "KernelAverage", "KernelStd")


def derived_fields(meshoid, field=None, name="f", which=("Density", "Gradient"), gradient_order=1):
    """Per-particle derived quantities from meshoid; returns {array name: (N, ...) array}.

    NB: anything needing neighbours makes meshoid recompute (and overwrite) its kernel
    radius/density/volume with its own des_ngb estimate, so these come out under
    Meshoid* names, distinct from GIZMO's SmoothingLength/Density.
    """
    M, out = meshoid, {}
    need_field = {"Gradient", "DivCurl", "KernelAverage", "KernelStd"} & set(which)
    f = None if field is None else np.asarray(field, dtype=np.float64)
    if need_field and f is None:
        raise ValueError(f"{sorted(need_field)} need a field")
    if "Gradient" in which:
        g = M.D(f, order=gradient_order)
        out[f"grad_{name}"] = g.reshape(len(g), -1)  # vector field: 9 comps, [i*3 + k] = d f_i / d x_k
    if "DivCurl" in which:
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("divergence/curl need a 3-component vector field")
        out[f"div_{name}"] = M.Div(f)
        out[f"curl_{name}"] = M.Curl(f)
    if "KernelAverage" in which:
        # meshoid's KernelAverage is scalar-only (einsum "ij,ij->i"): do vectors per component
        out[f"KernelAverage_{name}"] = (M.KernelAverage(f) if f.ndim == 1 else
                                         np.stack([M.KernelAverage(f[:, k]) for k in range(f.shape[1])], 1))
    if "KernelStd" in which:
        out[f"KernelStd_{name}"] = M.KernelVariance(f)  # meshoid returns the std despite the name
    # after any neighbour access these are meshoid's own (des_ngb) estimates
    if "KernelRadius" in which:
        out["MeshoidKernelRadius"] = np.asarray(M.SmoothingLength(), dtype=np.float64)
    if "Density" in which:
        out["MeshoidDensity"] = np.asarray(M.Density(), dtype=np.float64)
    if "Volume" in which:
        out["MeshoidVolume"] = np.asarray(M.Volume(), dtype=np.float64)
    return out
