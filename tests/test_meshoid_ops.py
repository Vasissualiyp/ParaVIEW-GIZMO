import numpy as np
import pytest
from meshoid import Meshoid

import snapgen
from pvgizmo import meshoid_ops as mo
from pvgizmo import vtkconv

RES = 32


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(42)
    pos = snapgen.plummer(3000, rng=rng)
    m = rng.uniform(0.5, 1.5, 3000) / 3000
    M = Meshoid(pos, m)
    H = M.SmoothingLength()
    q = rng.normal(size=3000) + pos[:, 0]  # intensive field with structure
    vel = rng.normal(size=(3000, 3))
    return dict(pos=pos, m=m, H=H, M=Meshoid(pos, m, kernel_radius=H), q=q, vel=vel)


def test_view_basis():
    np.testing.assert_allclose(mo.view_basis((0, 0, 1)), np.eye(3), atol=1e-15)
    for n, up in [((1, 2, 3), (0, 1, 0)), ((0, 1, 0), (0, 1, 0)), ((0, 0, -1), (1, 1, 0))]:
        B = mo.view_basis(n, up)
        np.testing.assert_allclose(B @ B.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(B) == pytest.approx(1.0)
        np.testing.assert_allclose(B[2], np.array(n) / np.linalg.norm(n))
    with pytest.raises(ValueError):
        mo.view_basis((0, 0, 0))


CENTER, SIZE = np.array([0.1, -0.2, 0.05]), 3.0


@pytest.mark.parametrize("mode", mo.MODES)
def test_z_projection_matches_meshoid(cloud, mode):
    M = cloud["M"]
    args = dict(size=SIZE, center=CENTER, res=RES)
    if mode == "SurfaceDensity":
        ref = M.SurfaceDensity(M.m, **args)
        out = mo.project(mode, cloud["pos"], cloud["H"], CENTER, SIZE, RES, field=cloud["m"])
    elif mode == "LineIntegral":
        ref = M.Projection(cloud["q"], **args)
        out = mo.project(mode, cloud["pos"], cloud["H"], CENTER, SIZE, RES, field=cloud["q"], volume=M.vol)
    elif mode == "MassWeightedAverage":
        ref = M.MassWeightedAverage(cloud["q"], **args)
        out = mo.project(mode, cloud["pos"], cloud["H"], CENTER, SIZE, RES, field=cloud["q"], masses=cloud["m"])
    else:
        ref = M.ProjectedAverage(cloud["q"], **args)
        out = mo.project(mode, cloud["pos"], cloud["H"], CENTER, SIZE, RES, field=cloud["q"])
    assert out.shape == (RES, RES)
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-14 * np.nanmax(np.abs(ref)))


def test_rotated_view_equals_permuted_data(cloud):
    """Looking along +x with up=z: image axes (u, v) = (y, z)."""
    pos, H, m = cloud["pos"], cloud["H"], cloud["m"]
    out = mo.project("SurfaceDensity", pos, H, CENTER, SIZE, RES, normal=(1, 0, 0), view_up=(0, 0, 1), field=m)
    perm = pos[:, [1, 2, 0]]
    ref = mo.project("SurfaceDensity", perm, H, CENTER[[1, 2, 0]], SIZE, RES, field=m)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_conservative_mass_any_orientation(cloud):
    """Conservative deposition keeps the mass inside the window, whatever the normal."""
    pos, H, m = cloud["pos"], cloud["H"], cloud["m"]
    size, res = 40.0, 64
    dA = mo.pixel_area(size, res)  # node grid: spacing size/(res-1)
    rng = np.random.default_rng(0)
    Hc = np.maximum(H, 2 * size / res)  # project() clips kernels to >= 2 pixels
    for n in [(0, 0, 1)] + list(rng.normal(size=(3, 3))):
        x = mo.to_view(pos, np.zeros(3), mo.view_basis(n))
        inside = np.all(np.abs(x[:, :2]) + Hc[:, None] < size / 2, axis=1)  # whole kernel in window
        assert inside.sum() > 2500
        sd = mo.project("SurfaceDensity", pos[inside], H[inside], np.zeros(3), size, res, normal=n,
                        field=m[inside], conservative=True)
        assert sd.sum() * dA == pytest.approx(m[inside].sum(), rel=1e-10)


def test_vector_field_and_depth(cloud):
    pos, H, m, vel = cloud["pos"], cloud["H"], cloud["m"], cloud["vel"]
    out = mo.project("MassWeightedAverage", pos, H, CENTER, SIZE, RES, field=vel, masses=m)
    assert out.shape == (RES, RES, 3)
    one = mo.project("MassWeightedAverage", pos, H, CENTER, SIZE, RES, field=vel[:, 1], masses=m)
    np.testing.assert_allclose(out[..., 1], one, rtol=1e-12)
    full = mo.project("SurfaceDensity", pos, H, CENTER, SIZE, RES, field=m)
    slab = mo.project("SurfaceDensity", pos, H, CENTER, SIZE, RES, field=m, depth=0.5)
    keep = np.abs(pos[:, 2] - CENTER[2]) < 0.25
    ref = mo.project("SurfaceDensity", pos[keep], H[keep], CENTER, SIZE, RES, field=m[keep])
    np.testing.assert_allclose(slab, ref, rtol=1e-12)
    assert slab.sum() < full.sum()


def test_z_slice_matches_meshoid_reconstruct(cloud):
    """Same as meshoid's Slice, but on the node grid shared with projections."""
    M = cloud["M"]
    s = mo.grid_coords(SIZE, RES)
    X, Y = np.meshgrid(s, s, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), np.zeros(RES * RES)] + CENTER
    ref = M.Reconstruct(cloud["q"], pts, order=0).reshape(RES, RES)
    np.testing.assert_array_equal(mo.slice_field(M, cloud["q"], CENTER, SIZE, RES), ref)


def test_projection_node_grid_geometry():
    """A compact particle lands on the node nearest to it, wherever the view points."""
    for n, up in [((0, 0, 1), (0, 1, 0)), ((1, 1, 0), (0, 0, 1))]:
        origin, spacing, direction = mo.image_geometry(np.zeros(3), 2.0, 21, n, up)
        target = origin + direction @ np.array([13 * spacing[0], 4 * spacing[1], 0.0])
        sd = mo.project("SurfaceDensity", target[None], np.array([1e-3]), np.zeros(3), 2.0, 21,
                        normal=n, view_up=up, field=np.ones(1))
        assert np.unravel_index(np.argmax(sd), sd.shape) == (13, 4)


def test_oblique_slice_reconstructs_linear_field_exactly(cloud):
    """Least-squares gradients are exact for linear fields, so order-1 slices are too."""
    M = mo.MeshoidCache().get("k", cloud["pos"], cloud["m"], cloud["H"])
    a, b = np.array([0.3, -1.2, 2.0]), 0.7
    f = cloud["pos"] @ a + b
    n, up = (1, 1, 1), (0, 0, 1)
    out = mo.slice_field(M, f, CENTER, 1.0, 16, normal=n, view_up=up, order=1)
    pts = mo.slice_points(CENTER, 1.0, 16, n, up)
    np.testing.assert_allclose(out.ravel(), pts @ a + b, rtol=1e-9, atol=1e-9)


def test_image_geometry_matches_slice_points():
    n, up = (1, -2, 0.5), (0, 0, 1)
    origin, spacing, direction = mo.image_geometry(CENTER, 2.0, 8, n, up)
    img = vtkconv.image_from_array(np.zeros((8, 8)), origin, spacing, direction)
    pts = mo.slice_points(CENTER, 2.0, 8, n, up)
    for i, j in [(0, 0), (7, 0), (3, 5)]:
        np.testing.assert_allclose(img.GetPoint(img.ComputePointId([i, j, 0])), pts[i * 8 + j], atol=1e-12)


def test_cache_lru(cloud):
    c = mo.MeshoidCache(maxsize=2)
    a = c.get("a", cloud["pos"])
    assert c.get("a", cloud["pos"]) is a
    c.get("b", cloud["pos"])
    c.get("c", cloud["pos"])
    assert c.get("a", cloud["pos"]) is not a  # evicted


def test_auto_extent(cloud):
    c, size = mo.auto_extent(cloud["pos"])
    assert np.allclose(c, np.median(cloud["pos"], axis=0)) and size > 0
