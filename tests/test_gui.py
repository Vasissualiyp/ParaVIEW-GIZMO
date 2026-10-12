"""Run the real ParaView GUI headless (Xvfb) on the reader; checks it renders."""

import os
import shutil
import signal
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def n_colors(png):
    from vtkmodules.util import numpy_support as ns
    from vtkmodules.vtkIOImage import vtkPNGReader
    r = vtkPNGReader()
    r.SetFileName(png)
    r.Update()
    px = ns.vtk_to_numpy(r.GetOutput().GetPointData().GetScalars())
    return len(np.unique(px, axis=0))


@pytest.mark.skipif(not (shutil.which("xvfb-run") and shutil.which("paraview")),
                    reason="needs xvfb-run and paraview (nix develop)")
def test_gui_smoke(snapdir, tmp_path):
    shot = str(tmp_path / "gui.png")
    # Isolated settings: no user config leaks in, and ParaView 6's modal Welcome
    # dialog (opens on a timer, blocks the script under Xvfb) is switched off.
    cfg = tmp_path / "config"
    (cfg / "ParaView").mkdir(parents=True)
    (cfg / "ParaView" / "ParaView.ini").write_text("[General]\nGeneralSettings.ShowWelcomeDialog=false\n")
    env = dict(os.environ, SNAPDIR=snapdir, SHOT=shot, XDG_CONFIG_HOME=str(cfg))
    # own session so a hung GUI (and its Xvfb) can be killed as a group
    p = subprocess.Popen(["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", "paraview",
                          f"--script={HERE}/gui_smoke.py", "--exit"], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        log, _ = p.communicate()
        pytest.fail("ParaView GUI hung:\n" + log[-3000:])
    assert "GUI_SMOKE_OK 4000" in log, log[-3000:]
    assert "Traceback" not in log and "NumbaSystemWarning" not in log, log[-3000:]
    proj_shot = shot.replace(".png", "_proj.png")
    vol_shot = shot.replace(".png", "_volume.png")
    for png in (shot, proj_shot, vol_shot):  # something was rendered, not a blank view
        assert n_colors(png) > 50, png
    if os.environ.get("PV_GIZMO_KEEP_SHOT"):
        keep = os.environ["PV_GIZMO_KEEP_SHOT"]
        shutil.copy(shot, keep)
        shutil.copy(proj_shot, keep.replace(".png", "_proj.png"))
        shutil.copy(vol_shot, keep.replace(".png", "_volume.png"))


@pytest.mark.skipif(not (shutil.which("xvfb-run") and shutil.which("paraview") and shutil.which("xdotool")),
                    reason="needs xvfb-run, paraview and xdotool (nix develop)")
def test_live_mouse_drag(snapdir, tmp_path):
    """A real mouse drag in the Qt GUI re-aims a FollowCamera projection, live.

    Checks the whole chain: Qt mouse events -> ParaView camera -> pvgizmo.live link
    (auto-installed for FollowCamera=1) -> filter re-executes at the interactive
    resolution while dragging -> one full-resolution run after release.
    """
    import json
    cfg = tmp_path / "config"
    (cfg / "ParaView").mkdir(parents=True)
    (cfg / "ParaView" / "ParaView.ini").write_text("[General]\nGeneralSettings.ShowWelcomeDialog=false\n")
    log = tmp_path / "live.log"
    env = dict(os.environ, XDG_CONFIG_HOME=str(cfg), PV_GIZMO_LIVE_LOG=str(log),
               SNAPFILE=f"{snapdir}/single/snapshot_000.hdf5")
    p = subprocess.Popen(["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", f"{HERE}/gui_drag.sh"], env=env,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        pytest.fail("GUI drag test hung:\n" + out[-3000:])
    assert "GUI_DRAG_DONE" in out, out[-3000:]
    ev = [json.loads(line) for line in log.read_text().splitlines()]
    ready = next(e for e in ev if e["event"] == "ready")
    assert ready["auto_linked"] and ready["gui"], ready
    i0, i1 = [e["event"] for e in ev].index("start"), [e["event"] for e in ev].index("end")
    during = [e for e in ev[i0:i1] if e["event"] == "exec"]
    after = [e for e in ev[i1:] if e["event"] == "exec"]
    assert len(during) >= 5 and all(e["res"] == 128 for e in during), during  # live, low-res
    assert after and after[-1]["res"] == 200  # full resolution after release
    n0 = np.array(next(e for e in ev if e["event"] == "exec")["normal"])
    n1 = np.array(after[-1]["normal"])
    assert np.dot(n0, n1) < 0.9, (n0, n1)  # the view really rotated the projection
