"""GUI smoke test, run inside the ParaView client:

    xvfb-run -a paraview --script=tests/gui_smoke.py --exit      (env: SNAPDIR, SHOT)

Needs GeneralSettings.ShowWelcomeDialog=false (test_gui.py writes an isolated
XDG_CONFIG_HOME), otherwise the modal Welcome dialog blocks the run.

Writes "GUI_SMOKE_OK" to stdout on success; tests/test_gui.py drives it.
"""

import os
import sys
import traceback

from paraview.simple import (ColorBy, ExtractBlock, GetActiveViewOrCreate, GetAnimationScene,
                             GetColorTransferFunction, Render, ResetCamera, SaveScreenshot, Show)
import paraview.simple as pvs


# An uncaught exception in a GUI script is shown in ParaView's output window and
# --exit never fires, so report it on the terminal and hard-exit instead.
try:
    snapdir = os.environ["SNAPDIR"]
    files = [f"{snapdir}/series/snapshot_{i:03d}.hdf5" for i in range(3)]
    r = pvs.GIZMOSnapshotReader(FileNames=files)
    r.UpdatePipelineInformation()
    assert list(r.TimestepValues) == [0.0, 0.1, 0.2], r.TimestepValues
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.AnimationTime = 0.2
    r.UpdatePipeline(0.2)  # data information is empty until the pipeline has run

    info = r.GetDataInformation().DataInformation
    names = {info.GetHierarchy().GetAttributeOrDefault(n, "label", "")
             for n in info.GetHierarchy().GetChildNodes(0)}
    assert {"Gas", "DarkMatter", "Stars"} <= names, names

    assert r.FieldData["Time"].GetRange() == (0.2, 0.2), r.FieldData["Time"].GetRange()

    gas = ExtractBlock(Input=r, Selectors=["/Root/Gas"])
    gas.UpdatePipeline(0.2)
    assert gas.GetDataInformation().GetNumberOfPoints() == 4000, gas.GetDataInformation().GetNumberOfPoints()
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [800, 600]
    rep = Show(gas, view)
    rep.SetRepresentationType("Point Gaussian")
    rep.ScaleByArray = 1
    rep.ShaderPreset = "Gaussian Blur"
    rep.SetScaleArray = ["POINTS", "SmoothingLengthVTK"]  # H/2: splat ~ kernel core
    rep.UseScaleFunction = 0
    rep.GaussianRadius = 1.0
    rep.Opacity = 0.3
    ColorBy(rep, ("POINTS", "Density"))
    lut = GetColorTransferFunction("Density")
    lut.UseLogScale = 1
    rep.RescaleTransferFunctionToDataRange(False, True)
    ResetCamera(view)
    Render(view)
    SaveScreenshot(os.environ["SHOT"], view, ImageResolution=[800, 600])
    print("GUI_SMOKE_OK", gas.GetDataInformation().GetNumberOfPoints(), flush=True)
except Exception:
    # the GUI redirects sys.stderr into its output window; use the real stdout
    traceback.print_exc(file=sys.__stdout__)
    sys.__stdout__.flush()
    os._exit(1)
