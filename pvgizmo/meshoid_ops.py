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
from meshoid.grid_deposition import GridAverage, GridSurfaceDensityMulti

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
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    basis = view_basis(normal, view_up)
    x = to_view(pos, center, basis)
    h = np.clip(smooth_fac * np.asarray(h, dtype=np.float64), 2.0 * size / res, None)
    sel = np.abs(x[:, 2]) < depth / 2 if depth > 0 else slice(None)
    x, h = x[sel], h[sel]
    origin = np.zeros(3)
    kw = dict(box_size=-1, parallel=parallel)
    if field is not None:
        f, tail = _channels(field)
        f = f[sel]
    if mode == "SurfaceDensity":
        if field is None:
            raise ValueError("SurfaceDensity needs a field (e.g. Masses)")
        if conservative:  # conservative deposition exists for the single-channel routine only
            from meshoid.grid_deposition import GridSurfaceDensity
            out = np.stack([GridSurfaceDensity(f[:, k], x, h, origin, size, res, conservative=True, **kw)
                            for k in range(f.shape[1])], axis=-1)
        else:
            out = GridSurfaceDensityMulti(f, x, h, origin, size, res, **kw)
    elif mode == "LineIntegral":
        vol = np.asarray(volume, dtype=np.float64)[sel]
        out = GridSurfaceDensityMulti(f * vol[:, None], x, h, origin, size, res, **kw)
    elif mode == "MassWeightedAverage":
        m = np.asarray(masses, dtype=np.float64)[sel]
        stacked = GridSurfaceDensityMulti(np.column_stack([f * m[:, None], m]), x, h, origin, size, res, **kw)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = stacked[..., :-1] / stacked[..., -1:]
    else:  # ProjectedAverage
        out = np.stack([GridAverage(f[:, k], x, h, origin, size, res, parallel=parallel)
                        for k in range(f.shape[1])], axis=-1)
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
