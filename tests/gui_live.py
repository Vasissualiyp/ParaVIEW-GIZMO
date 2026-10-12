"""Setup for the GUI interactivity test (tests/test_gui.py::test_live_mouse_drag).

Runs inside `paraview --script=...` *without* --exit: it builds the scene and
returns, so the Qt event loop runs and real (xdotool) mouse drags reach the view.
"""

import os
import sys
import traceback

try:
    import paraview.simple as pvs
    from pvgizmo import live, pvhelpers

    r = pvhelpers.open_snapshot(os.environ["SNAPFILE"])
    view = pvs.GetActiveViewOrCreate("RenderView")
    proj = pvs.MeshoidProjection(Input=r, Resolution=200)  # FollowCamera defaults to on
    pvhelpers.show_map(proj, view)
    pvs.ResetCamera(view)
    pvs.Render(view)
    live.debug_log("ready", auto_linked=live.following(proj), gui=live.in_gui(), installed=live._AUTO["installed"], now=live.enable_auto_follow())
    print("GUI_LIVE_READY", flush=True)
except Exception:
    traceback.print_exc(file=sys.__stdout__)
    sys.__stdout__.flush()
    os._exit(1)
