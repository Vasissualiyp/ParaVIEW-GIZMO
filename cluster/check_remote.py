"""Headless end-to-end check against the Trillium pvserver (through cluster/connect.sh's tunnel).

    cluster/connect.sh --tunnel-only
    nix develop -c pvpython cluster/check_remote.py [SNAPSHOT] [--port 11111] [--out DIR]

The snapshot path is on the cluster (the server opens it). Opens it, projects the gas
column density, builds the space look, then centres on the most massive star and zooms
through a dense-gas volume, timing each step, and saves screenshots.
Prints REMOTE_OK on success.
"""

import argparse
import os
import time

import paraview.simple as pvs
from paraview import servermanager as sm

DEFAULT = "/scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout/snapshot_259.hdf5"
ap = argparse.ArgumentParser()
ap.add_argument("snapshot", nargs="?", default=DEFAULT)
ap.add_argument("--port", type=int, default=11111)
ap.add_argument("--out", default=".")
a = ap.parse_args()


def step(msg, t0=[time.perf_counter()]):
    t = time.perf_counter()
    print(f"[{t - t0[0]:7.2f} s] {msg}", flush=True)
    t0[0] = t


pvs.Connect("localhost", a.port)
assert sm.ActiveConnection.IsRemote(), "not connected to a remote server"
for name in ("GIZMOSnapshotReader", "MeshoidProjection", "GIZMOStarlightImage", "MeshoidComposite"):
    assert hasattr(pvs, name), f"{name} missing: plugins not loaded on both sides"
step(f"connected to localhost:{a.port}")

from pvgizmo import live, pvhelpers  # noqa: E402

r = pvhelpers.open_snapshot(a.snapshot, Recenter=1)
step(f"read {a.snapshot}: {r.GetDataInformation().GetNumberOfPoints()} points, blocks {pvhelpers.block_names(r)}")

view = pvs.CreateRenderView()
view.ViewSize = [800, 800]
view.OrientationAxesVisibility = 0
p = pvhelpers.add_projection(r, view=view, Resolution=512)
pvs.ResetCamera(view)
live.follow_camera(p, view)
pvs.Render(view)
step("gas projection 512^2 (first call includes numba JIT on the server)")
for az in (30, 60):
    view.GetActiveCamera().Azimuth(az)
    pvs.Render(view)
    step(f"rotated {az} deg: re-projected and delivered")
pvs.SaveScreenshot(os.path.join(a.out, "remote_projection.png"), view, ImageResolution=[800, 800])
pvs.Hide(p, view)

made = pvhelpers.space_look(r, view)
for f in made.values():  # pvpython has no GUI auto-follow: link explicitly
    live.follow_camera(f, view)
pvs.Render(view)
step(f"space look: {sorted(made)}")
pvs.SaveScreenshot(os.path.join(a.out, "remote_space.png"), view, ImageResolution=[800, 800])
view.GetActiveCamera().Zoom(4.0)
pvs.Render(view)
step("zoomed 4x: re-projected")
pvs.SaveScreenshot(os.path.join(a.out, "remote_space_zoom.png"), view, ImageResolution=[800, 800])
for f in made.values():
    live.unfollow(f)
    pvs.Hide(f, view)

pvhelpers.center_on_star(r)
step(f"recentred on star {r.FieldData['CenterParticleID'].GetRange()[0]:.0f}")
pvhelpers.star_camera(view, size=1e-2)
g = pvhelpers.add_dense_volume(r, view)
step("dense-gas volume (96^3, delivered and rendered on the client)")
for k in range(3):
    pvhelpers.zoom(10.0, view)
    step(f"zoomed to {live.view_window(view)[3]:.1e}: re-gridded, grid size {g.Size:.1e}")
pvs.SaveScreenshot(os.path.join(a.out, "remote_dense.png"), view, ImageResolution=[800, 800])
print("REMOTE_OK", flush=True)
