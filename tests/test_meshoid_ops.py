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


# ----------------------------------------------------------------------------- Phase 3
def test_grid3d_deposit_conserves_and_aligns_with_2d(cloud):
    pos, H, m = cloud["pos"], cloud["H"], cloud["m"]
    size, res = 40.0, 41
    dx = size / (res - 1)
    inside = np.all(np.abs(pos) + np.maximum(H, dx)[:, None] < size / 2 - dx, axis=1)
    rho = mo.grid3d_deposit(pos[inside], H[inside], m[inside], np.zeros(3), size, res)
    assert rho.shape == (res, res, res)
    assert rho.sum() * dx**3 == pytest.approx(m[inside].sum(), rel=1e-10)
    # a compact particle on node (k, j, l) lands on that node, like the 2D maps
    nodes = mo.grid_coords(2.0, 21)
    x = np.array([[nodes[5], nodes[13], nodes[8]]])
    g = mo.grid3d_deposit(x, np.array([0.25]), np.ones(1), np.zeros(3), 2.0, 21)
    c = [(g.sum(axis=tuple(a for a in range(3) if a != ax)) * np.arange(21)).sum() / g.sum() for ax in range(3)]
    np.testing.assert_allclose(c, [5, 13, 8], atol=1e-6)
    sd = mo.project("SurfaceDensity", x, np.array([0.25]), np.zeros(3), 2.0, 21, field=np.ones(1))
    assert np.unravel_index(np.argmax(sd), sd.shape) == (5, 13)


def test_grid3d_vector_deposit(cloud):
    pos, H, m, vel = cloud["pos"], cloud["H"], cloud["m"], cloud["vel"]
    mom = mo.grid3d_deposit(pos, H, m[:, None] * vel, np.zeros(3), 6.0, 17)
    one = mo.grid3d_deposit(pos, H, m * vel[:, 2], np.zeros(3), 6.0, 17)
    assert mom.shape == (17, 17, 17, 3)
    np.testing.assert_allclose(mom[..., 2], one, rtol=1e-12)


def test_grid3d_interp_linear_exact(cloud):
    M = mo.MeshoidCache().get("g", cloud["pos"], cloud["m"], cloud["H"])
    a, b = np.array([0.3, -1.2, 2.0]), 0.7
    f = cloud["pos"] @ a + b
    g = mo.grid3d_interp(M, f, CENTER, 1.0, 9, order=1)
    pts = mo.grid3d_points(CENTER, 1.0, 9)
    np.testing.assert_allclose(g.ravel(), pts @ a + b, rtol=1e-9, atol=1e-9)
    origin, spacing = mo.grid3d_geometry(CENTER, 1.0, 9)
    np.testing.assert_allclose(pts[0], origin)
    np.testing.assert_allclose(pts[9 * 9 + 9 + 1] - pts[0], spacing)  # (1,1,1) node


def test_derived_fields_exact_for_linear_fields(cloud):
    pos = cloud["pos"]
    M = Meshoid(pos, cloud["m"])
    a = np.array([0.3, -1.2, 2.0])
    A = np.array([[1.0, 2.0, 0.0], [-0.5, 0.3, 4.0], [0.2, 0.0, -1.0]])
    d = mo.derived_fields(M, pos @ a, "phi", which=("Gradient", "KernelAverage", "KernelStd", "Density"))
    np.testing.assert_allclose(d["grad_phi"], np.broadcast_to(a, (len(pos), 3)), rtol=1e-8, atol=1e-8)
    assert set(d) == {"grad_phi", "KernelAverage_phi", "KernelStd_phi", "MeshoidDensity"}
    v = pos @ A.T
    dv = mo.derived_fields(M, v, "v", which=("Gradient", "DivCurl", "KernelAverage"))
    np.testing.assert_allclose(dv["div_v"], np.trace(A), rtol=1e-8, atol=1e-8)
    curl = [A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]]
    np.testing.assert_allclose(dv["curl_v"], np.broadcast_to(curl, (len(pos), 3)), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(dv["grad_v"].reshape(-1, 3, 3), np.broadcast_to(A, (len(pos), 3, 3)),
                               rtol=1e-8, atol=1e-8)  # [i, k] = d v_i / d x_k
    assert dv["KernelAverage_v"].shape == (len(pos), 3)
    with pytest.raises(ValueError):
        mo.derived_fields(M, pos @ a, "phi", which=("DivCurl",))


def test_deposit3d_matches_meshoid_inside(cloud):
    """For kernels inside the box, pvgizmo.deposit == meshoid's GridDensity (node-shifted)."""
    from meshoid.grid_deposition import GridDensity
    pos, H, m = cloud["pos"], cloud["H"], cloud["m"]
    size, res = 8.0, 33
    dx = size / (res - 1)
    h = np.clip(H, dx, None)
    inside = np.all(np.abs(pos) + h[:, None] < size / 2, axis=1) & (h < 15 * dx)
    ours = mo.grid3d_deposit(pos[inside], H[inside], m[inside], np.zeros(3), size, res)
    ref = GridDensity(m[inside], pos[inside], h[inside], np.zeros(3) - dx / 2, size, res, box_size=-1)
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12 * ref.max())


def test_deposit3d_big_and_crossing_kernels():
    """Edge-crossing kernels deposit only their in-box share; no giant allocations."""
    from meshoid.kernel_density import kernel3d
    size, res = 1.0, 17
    dx = size / (res - 1)
    # a huge kernel (1e4 voxels) centred far outside: its density in the box is ~ m W(q)/H^3
    x = np.array([[300.0, 0.0, 0.0]])
    H = np.array([400.0])
    g = mo.grid3d_deposit(x, H, np.ones(1), np.zeros(3), size, res)
    q = np.linalg.norm(mo.grid3d_points(np.zeros(3), size, res) - x, axis=1) / 400.0
    want = np.array([kernel3d(v) for v in q]).reshape(g.shape) / 400.0**3
    np.testing.assert_allclose(g, want, rtol=1e-6)  # analytic normalisation for huge kernels
    assert g.sum() * dx**3 < 1e-6  # not the whole unit mass dumped into the box
    # wide (analytic-normalised) kernels fully inside still conserve mass closely
    x = np.zeros((1, 3))
    for r_vox in (20.0, 7.3):
        size, res = 2.0, 129
        dx = size / (res - 1)
        g = mo.grid3d_deposit(x, np.array([r_vox * dx]), np.ones(1), np.zeros(3), size, res)
        assert g.sum() * dx**3 == pytest.approx(1.0, rel=1e-4 if r_vox > 15 else 1e-12)


def test_conservative_projection_with_crossing_kernels(cloud):
    pos, H, m = cloud["pos"], cloud["H"], cloud["m"]
    size, res = 1.0, 32  # small window: many kernels cross the edge
    cons = mo.project("SurfaceDensity", pos, H, np.zeros(3), size, res, field=m, conservative=True)
    plain = mo.project("SurfaceDensity", pos, H, np.zeros(3), size, res, field=m)
    # no edge pile-up: the conservative map stays close to the plain one everywhere
    np.testing.assert_allclose(cons, plain, rtol=0.25)
    assert cons.sum() == pytest.approx(plain.sum(), rel=0.05)
