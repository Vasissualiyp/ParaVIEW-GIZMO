"""pvgizmo.live: projections that follow the camera (pvpython; the GUI test drives a real mouse)."""

import os

import numpy as np
import pytest

pvs = pytest.importorskip("paraview.simple")

from pvgizmo import live  # noqa: E402

PLUGINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin")


@pytest.fixture(scope="module", autouse=True)
def plugins():
    for name, f in (("GIZMOSnapshotReader", "GIZMOReader.py"), ("MeshoidProjection", "MeshoidFilters.py")):
        if not hasattr(pvs, name):
            pvs.LoadPlugin(os.path.join(PLUGINS, f), remote=False)


@pytest.fixture
def scene(snapdir):
    r = pvs.GIZMOSnapshotReader(FileNames=[f"{snapdir}/single/snapshot_000.hdf5"])
    r.UpdatePipeline()
    view = pvs.CreateRenderView()
    view.ViewSize = [400, 300]
    p = pvs.MeshoidProjection(Input=r, Resolution=64)
    pvs.Show(p, view)
    pvs.ResetCamera(view)
    yield r, view, p
    live.stop_all()
    for s in list(pvs.GetSources().values()):
        pvs.Delete(s)
    pvs.Delete(view)


def image_plane(p):
    p.UpdatePipeline()
    img = p.GetClientSideObject().GetOutputDataObject(0)
    m = img.GetDirectionMatrix()
    normal = np.array([m.GetElement(i, 2) for i in range(3)])
    nx, ny, _ = img.GetDimensions()
    center = np.array(img.GetPoint(img.ComputePointId([nx // 2, ny // 2, 0])))
    return normal, center, img


def test_not_auto_in_pvpython():
    assert not live.in_gui()  # scripts keep explicit windows; follow_camera() is opt-in here


def test_follows_rotation_zoom_pan(scene):
    _, view, p = scene
    live.follow_camera(p, view)
    cam = view.GetActiveCamera()
    for move in (lambda: cam.Azimuth(50), lambda: cam.Elevation(-30), lambda: cam.Zoom(2.0),
                 lambda: cam.SetFocalPoint(0.3, -0.2, 0.1)):
        move()  # ModifiedEvent -> re-aim before the next render
        center, normal, up, size = live.view_window(view)
        np.testing.assert_allclose(p.Normal, normal, atol=1e-12)
        np.testing.assert_allclose(p.Center, center, atol=1e-12)
        assert p.Size == pytest.approx(size)
        pvs.Render(view)
        n_img, c_img, _ = image_plane(p)
        np.testing.assert_allclose(n_img, normal, atol=1e-12)  # image faces the camera
        np.testing.assert_allclose(c_img, center, atol=size / 63)  # centred on the focal point


def test_window_covers_viewport(scene):
    _, view, p = scene
    live.follow_camera(p, view)
    cam = view.GetActiveCamera()
    d = np.linalg.norm(np.subtract(cam.GetPosition(), cam.GetFocalPoint()))
    height = 2 * d * np.tan(np.radians(cam.GetViewAngle() / 2))
    assert p.Size == pytest.approx(height * 400 / 300)  # wider than tall: cover the width
    view.CameraParallelProjection = 1
    cam.SetParallelScale(2.5)
    assert p.Size == pytest.approx(5.0 * 400 / 300)


def test_interactive_resolution_and_release(scene):
    _, view, p = scene
    link = live.follow_camera(p, view, interactive_resolution=16)
    view.SMProxy.InvokeEvent("StartInteractionEvent")
    view.GetActiveCamera().Azimuth(10)
    assert p.Resolution == 16
    pvs.Render(view)
    assert image_plane(p)[2].GetDimensions()[:2] == (16, 16)
    view.SMProxy.InvokeEvent("EndInteractionEvent")
    assert p.Resolution == 64 and not link._interacting
    assert image_plane(p)[2].GetDimensions()[:2] == (64, 64)


def test_ignores_non_geometric_camera_changes(scene, monkeypatch):
    _, view, p = scene
    link = live.follow_camera(p, view)
    calls = []
    monkeypatch.setattr(link, "apply", lambda interactive=False: calls.append(interactive))
    cam = view.GetActiveCamera()
    for _ in range(3):
        cam.SetClippingRange(0.01 + 0.01 * _, 100)  # what every render does
        pvs.Render(view)
    assert calls == []
    cam.Azimuth(5)
    assert len(calls) == 1


def test_unfollow_and_deleted_filters(scene):
    r, view, p = scene
    live.follow_camera(p, view, interactive_resolution=8)
    view.SMProxy.InvokeEvent("StartInteractionEvent")
    view.GetActiveCamera().Azimuth(10)
    live.unfollow(p)
    assert not live.following(p) and p.Resolution == 64  # restored
    before = list(p.Normal)
    view.GetActiveCamera().Azimuth(40)
    assert list(p.Normal) == before
    q = pvs.MeshoidProjection(Input=r, Resolution=32)
    live.follow_camera(q, view)
    pvs.Delete(q)
    view.GetActiveCamera().Azimuth(10)  # must not raise on the deleted proxy
    assert not any(f is q for link in live._LINKS.values() for f in link.filters)


def test_slice_can_follow(scene):
    r, view, _ = scene
    s = pvs.MeshoidSlice(Input=r, Resolution=32)
    assert s.FollowCamera == 0
    live.follow_camera(s, view)
    view.GetActiveCamera().Elevation(35)
    n_img, c_img, _ = image_plane(s)
    center, normal, _, _ = live.view_window(view)
    np.testing.assert_allclose(n_img, normal, atol=1e-12)
