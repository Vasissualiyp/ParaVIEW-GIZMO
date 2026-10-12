import gc

import numpy as np
from vtkmodules.util import numpy_support as ns

from pvgizmo import vtkconv


def test_polydata_roundtrip():
    rng = np.random.default_rng(1)
    pos = rng.random((100, 3)).astype(np.float32)
    rho = rng.random(100)
    met = rng.random((100, 11)).astype(np.float32)
    ids = np.arange(100, dtype=np.uint64)
    pd = vtkconv.particles_to_polydata(pos, {"Density": rho, "Metallicity": met, "ParticleIDs": ids},
                                       {"Time": 0.25})
    del pos, rho, met, ids  # VTK must keep the buffers alive
    gc.collect()
    assert pd.GetNumberOfPoints() == 100 and pd.GetNumberOfCells() == 1
    assert pd.GetVerts().GetCellSize(0) == 100
    pdat = pd.GetPointData()
    assert pdat.GetArray("Metallicity").GetNumberOfComponents() == 11
    assert ns.vtk_to_numpy(pdat.GetArray("ParticleIDs"))[-1] == 99
    rng = np.random.default_rng(1)
    np.testing.assert_array_equal(ns.vtk_to_numpy(pd.GetPoints().GetData()), rng.random((100, 3)).astype(np.float32))
    assert pd.GetFieldData().GetArray("Time").GetValue(0) == 0.25


def test_empty_polydata():
    pd = vtkconv.particles_to_polydata(np.zeros((0, 3)))
    assert pd.GetNumberOfPoints() == 0


def test_image_layout():
    v = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    img = vtkconv.image_from_array(v, origin=(1, 2, 3), spacing=(0.5, 0.5, 1.0))
    assert img.GetDimensions() == (4, 3, 2)
    a = img.GetPointData().GetArray("values")
    for i, j, k in [(0, 0, 0), (3, 0, 0), (1, 2, 1)]:
        pid = img.ComputePointId([i, j, k])
        assert a.GetValue(pid) == v[i, j, k]
        np.testing.assert_allclose(img.GetPoint(pid), (1 + 0.5 * i, 2 + 0.5 * j, 3 + k))


def test_image_2d_vector_and_direction():
    v = np.random.default_rng(0).random((5, 4, 3))  # 2D image with 3-component values
    v3 = v[:, :, None, :]
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    img = vtkconv.image_from_array(v3, (0, 0, 0), (1, 1, 1), direction=rot, name="vel")
    a = img.GetPointData().GetArray("vel")
    assert img.GetDimensions() == (5, 4, 1) and a.GetNumberOfComponents() == 3
    pid = img.ComputePointId([2, 1, 0])
    np.testing.assert_allclose(a.GetTuple3(pid), v[2, 1])
    np.testing.assert_allclose(img.GetPoint(pid), rot @ [2, 1, 0])
