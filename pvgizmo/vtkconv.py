"""numpy -> VTK data objects. Zero-copy where VTK's layout allows it."""

import numpy as np
from vtkmodules.util import numpy_support as ns
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData


def to_vtk_array(arr, name):
    """Wrap a (N,) or (N,k) numpy array as a vtkDataArray named `name`.

    The VTK array keeps a reference to the (contiguous) numpy buffer, so the
    caller does not need to keep `arr` alive.
    """
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float16:
        a = a.astype(np.float32)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    v = ns.numpy_to_vtk(a, deep=False)
    v._numpy_ref = a  # numpy_to_vtk(deep=False) does not hold the buffer itself
    v.SetName(name)
    return v


def particles_to_polydata(pos, arrays=None, field_data=None):
    """Particles -> vtkPolyData with points and one poly-vertex cell.

    One poly-vertex cell (rather than N vertex cells) keeps memory at ~N ints
    and is what Point Gaussian and the point-based filters expect.
    """
    pd = vtkPolyData()
    n = len(pos)
    pts = vtkPoints()
    pts.SetData(to_vtk_array(pos, "Points"))
    pd.SetPoints(pts)
    if n:
        conn = np.arange(n, dtype=np.int64)
        offs = np.array([0, n], dtype=np.int64)
        cells = vtkCellArray()
        cells.SetData(to_vtk_array(offs, "offsets"), to_vtk_array(conn, "connectivity"))
        pd.SetVerts(cells)
    for name, a in (arrays or {}).items():
        pd.GetPointData().AddArray(to_vtk_array(a, name))
    add_field_data(pd, field_data or {})
    return pd


def add_field_data(obj, field_data):
    fd = obj.GetFieldData()
    for name, v in field_data.items():
        a = np.atleast_1d(np.asarray(v))
        if a.dtype.kind in "biuf":
            fd.AddArray(to_vtk_array(a.astype(np.float64), name))


def image_from_array(values, origin, spacing, direction=None, name="values"):
    """(nx, ny[, nz]) numpy array (x fastest *last* index convention: values[i, j, k]
    at origin + i*dx*u + j*dy*v + k*dz*w) -> vtkImageData with point data `name`.
    """
    v = np.asarray(values)
    if v.ndim == 2:
        v = v[:, :, None]
    dims = list(v.shape[:3])
    ncomp = int(np.prod(v.shape[3:])) if v.ndim > 3 else 1
    img = vtkImageData()
    img.SetDimensions(*dims)
    img.SetOrigin(*origin)
    img.SetSpacing(*spacing)
    if direction is not None:
        img.SetDirectionMatrix(*np.asarray(direction, dtype=np.float64).ravel())
    # VTK point order is i fastest, then j, then k
    flat = v.reshape(dims + [ncomp]).transpose(2, 1, 0, 3).reshape(-1, ncomp)
    if ncomp == 1:
        flat = flat[:, 0]
    img.GetPointData().AddArray(to_vtk_array(flat, name))
    img.GetPointData().SetActiveScalars(name)
    return img
