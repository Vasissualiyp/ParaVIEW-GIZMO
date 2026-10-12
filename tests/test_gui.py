"""Run the real ParaView GUI headless (Xvfb) on the reader; checks it renders."""

import os
import shutil
import signal
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


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
    assert os.path.getsize(shot) > 10_000
    if os.environ.get("PV_GIZMO_KEEP_SHOT"):
        shutil.copy(shot, os.environ["PV_GIZMO_KEEP_SHOT"])
