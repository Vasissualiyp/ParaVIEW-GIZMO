"""Phase 6 GUI check, run inside the ParaView client (see gui_smoke.py for the setup):

    xvfb-run -a paraview --script=tests/gui_space.py --exit      (env: SNAPDIR, SHOT)

1. FIRE-like galaxy: space look -> starlight image that follows the camera (rotated
   to edge-on by script: the auto-linked filter must re-aim).
2. STARFORGE-like sinks: glowing gas composite + glowing (additive Point Gaussian) stars.
3. Stars alone on black, for the bright-pixels-at-star-positions check.
Prints "GUI_SPACE_OK" on success.
"""

import json
import os
import sys
import traceback

import numpy as np
import paraview.simple as pvs

try:
    from pvgizmo import live, pvhelpers

    snapdir, shot = os.environ["SNAPDIR"], os.environ["SHOT"]
    view = pvs.GetActiveViewOrCreate("RenderView")
    view.ViewSize = [800, 600]

    gal = pvhelpers.open_snapshot(f"{snapdir}/galaxy/snapshot_600.hdf5")
    made = pvhelpers.space_look(gal, view)
    sl = made["starlight"]
    view.CameraPosition, view.CameraFocalPoint, view.CameraViewUp = [0, 0, 40], [0, 0, 0], [0, 1, 0]
    pvs.Render(view)
    assert live.following(sl), "starlight image not auto-linked to the camera"
    pvs.SaveScreenshot(shot.replace(".png", "_faceon.png"), view, ImageResolution=[800, 600])
    view.CameraPosition, view.CameraViewUp = [0, -40, 2], [0, 0, 1]  # edge-on
    pvs.Render(view)
    n = np.asarray(sl.Normal)
    assert abs(n[1]) > 0.99, f"starlight did not follow the camera: Normal={list(n)}"
    pvs.SaveScreenshot(shot.replace(".png", "_edgeon.png"), view, ImageResolution=[800, 600])
    for s in list(pvs.GetSources().values()):
        pvs.Hide(s, view)

    sinks = pvhelpers.open_snapshot(f"{snapdir}/sinks/snapshot_010.hdf5")
    made = pvhelpers.space_look(sinks, view)
    assert set(made) == {"composite", "stars"}, made
    pvs.ResetCamera(view)
    pvs.Render(view)
    pvs.SaveScreenshot(shot.replace(".png", "_sinks.png"), view, ImageResolution=[800, 600])

    pvs.Hide(made["composite"], view)
    cam = view.GetActiveCamera()
    cam.SetFocalPoint(0, 0, 0)
    cam.SetPosition(0, 0, 6)
    cam.SetViewUp(0, 1, 0)
    pvs.Render(view)
    pvs.SaveScreenshot(shot.replace(".png", "_stars.png"), view, ImageResolution=[800, 600])
    pvs.Show(made["composite"], view)  # stars must still show over the (opaque) gas map
    pvs.Render(view)
    pvs.SaveScreenshot(shot.replace(".png", "_overlay.png"), view, ImageResolution=[800, 600])
    pvs.Hide(made["composite"], view)
    # where the stars should be: project their positions with the camera matrices
    from paraview import servermanager
    from vtkmodules.util import numpy_support as ns
    sp = made["stars"]
    out = servermanager.Fetch(pvs.ExtractBlock(Input=sp, Selectors=["/Root/Type5"]))
    leaf = out
    while not hasattr(leaf, "GetPoints"):
        leaf = leaf.GetPartitionedDataSet(0).GetPartition(0) if hasattr(leaf, "GetPartitionedDataSet") \
            else leaf.GetPartition(0)
    pos = ns.vtk_to_numpy(leaf.GetPoints().GetData())
    ren = view.GetRenderer()
    px = []
    for p in pos:
        ren.SetWorldPoint(*p, 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        px.append([x, y])
    with open(shot.replace(".png", "_stars.json"), "w") as fh:
        json.dump({"pixels": px, "size": list(view.ViewSize)}, fh)
    print("GUI_SPACE_OK", flush=True)
except Exception:
    traceback.print_exc(file=sys.__stdout__)
    sys.__stdout__.flush()
    os._exit(1)
