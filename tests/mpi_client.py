"""Client of tests/test_client_server.py::test_mpi_pvserver (pvserver under mpiexec -n 3).

    pvpython tests/mpi_client.py PORT SNAPFILE
The results must equal the single-process ones: no data duplicated across ranks.
"""

import os
import sys

import numpy as np
import paraview.simple as pvs
from paraview import servermanager as sm
from vtkmodules.util import numpy_support as ns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pvgizmo import io  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402

port, snap = int(sys.argv[1]), sys.argv[2]
pvs.Connect("localhost", port)
assert sm.ActiveConnection.GetNumberOfDataPartitions() == 3

r = pvs.GIZMOSnapshotReader(FileNames=[snap])
r.UpdatePipeline()
assert r.GetDataInformation().GetNumberOfPoints() == 6500, r.GetDataInformation().GetNumberOfPoints()

d = io.read_ptype(io.snapshot_parts(snap), 0, ["Masses", "SmoothingLength"])
p = pvs.MeshoidProjection(Input=r, AutoWindow=0, Center=[0, 0, 0], Size=6.0, Resolution=32,
                          Normal=[1, 2, 3], ViewUp=[0, 0, 1])
p.UpdatePipeline()
assert p.GetDataInformation().GetNumberOfPoints() == 32 * 32
img = sm.Fetch(p)
v = ns.vtk_to_numpy(img.GetPointData().GetArray("SurfaceDensity_Masses"))
ref = mo.project("SurfaceDensity", d["Coordinates"], d["SmoothingLength"], np.zeros(3), 6.0, 32,
                 normal=(1, 2, 3), view_up=(0, 0, 1), field=d["Masses"])
assert np.isclose(v.sum(), ref.sum(), rtol=1e-12), (v.sum(), ref.sum())

s = pvs.MeshoidSlice(Input=r, Resolution=16)
s.UpdatePipeline()
assert s.GetDataInformation().GetNumberOfPoints() == 16 * 16
g = pvs.MeshoidGrid(Input=r, Resolution=8)
g.UpdatePipeline()
assert g.GetDataInformation().GetNumberOfPoints() == 8 ** 3
dv = pvs.MeshoidDerivedFields(Input=r)
dv.UpdatePipeline()
assert dv.GetDataInformation().GetNumberOfPoints() == 6500
assert "grad_Density" in dv.PointData.keys()
print("MPI_OK", flush=True)
