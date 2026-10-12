"""Client side of tests/test_client_server.py: run with pvpython against a live pvserver.

    pvpython tests/cs_client.py PORT SNAPFILE OUTDIR
Prints CS_OK on success, assertion tracebacks otherwise.
"""

import os
import sys

import numpy as np
import paraview.simple as pvs
from paraview import servermanager as sm

port, snap, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
pvs.Connect("localhost", port)
assert sm.ActiveConnection.IsRemote()
for name in ("GIZMOSnapshotReader", "MeshoidProjection", "MeshoidSlice", "MeshoidGrid", "MeshoidDerivedFields"):
    assert hasattr(pvs, name), name  # plugins loaded on both sides (PV_PLUGIN_PATH)

from pvgizmo import live, pvhelpers  # noqa: E402

r = pvhelpers.open_snapshot(snap)  # the path is opened by the server
assert r.GetDataInformation().GetNumberOfPoints() == 6500


def plane_normal(filt):
    sg = sm.Fetch(filt)  # data delivered to the client, as for local rendering
    e = sg.GetExtent()
    nx = e[1] - e[0] + 1
    p00, p10, p01 = (np.array(sg.GetPoint(k)) for k in (0, 1, nx))
    n = np.cross(p10 - p00, p01 - p00)
    return n / np.linalg.norm(n)


view = pvs.CreateRenderView()
view.ViewSize = [300, 300]
view.OrientationAxesVisibility = 0
p = pvs.MeshoidProjection(Input=r, Resolution=48)
pvs.Show(p, view)
pvs.ResetCamera(view)
live.follow_camera(p, view)
for az in (70, -30):
    view.GetActiveCamera().Azimuth(az)
    pvs.Render(view)
    want = live.view_window(view)[1]
    got = plane_normal(p)
    assert np.allclose(got, want, atol=1e-9), (got, want)  # orientation survives delivery

# edge-on check, rendered locally (delivered geometry) and remotely (server-side render)
live.unfollow(p)
p.AutoWindow, p.Center, p.Size, p.Normal, p.ViewUp = 0, [0, 0, 0], 6.0, [1, 0, 0], [0, 0, 1]
view.CameraPosition, view.CameraFocalPoint, view.CameraViewUp = [0, -20, 0], [0, 0, 0], [0, 0, 1]
for mode, threshold in (("local", 1e9), ("remote", 0)):
    view.RemoteRenderThreshold = threshold
    pvs.Render(view)
    png = os.path.join(outdir, f"cs_{mode}.png")
    pvs.SaveScreenshot(png, view, ImageResolution=[300, 300])
    from vtkmodules.util import numpy_support as ns
    from vtkmodules.vtkIOImage import vtkPNGReader
    rd = vtkPNGReader()
    rd.SetFileName(png)
    rd.Update()
    px = ns.vtk_to_numpy(rd.GetOutput().GetPointData().GetScalars())[:, :3]
    visible = np.any(px != px[0], axis=1).mean()
    assert visible < 0.002, (mode, visible)  # a plane seen edge-on must (almost) vanish

g = pvs.MeshoidGrid(Input=r, Resolution=16)
assert sm.Fetch(g).GetDimensions() == (16, 16, 16)
d = pvs.MeshoidDerivedFields(Input=r)
d.UpdatePipeline()
print("CS_OK", flush=True)
