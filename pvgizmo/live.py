"""Live, view-dependent Meshoid maps: projections/slices that follow the camera.

A CameraLink observes a render view's camera on the *client*. On every
geometric camera change (rotate, zoom, pan, toolbar buttons, scripted moves)
it re-aims its filters at the view before the frame renders: line of sight,
up, centre (focal point) and window size (the visible extent at the focal
plane). While the mouse is down the filters run at a reduced resolution
(`interactive_resolution`); on release they go back to their full resolution
and the view re-renders.

    from pvgizmo import live
    live.follow_camera(proj)          # proj = a Meshoid Projection / Slice proxy
    live.unfollow(proj)

Filters with FollowCamera = 1 are linked automatically when created in the
GUI (see enable_auto_follow, called by the plugin on the client).
"""

import numpy as np

_LINKS = {}  # id(view proxy) -> CameraLink


def debug_log(event, **kw):
    """Append a JSON line to $PV_GIZMO_LIVE_LOG (tests of GUI interactivity); no-op otherwise."""
    import json
    import os
    path = os.environ.get("PV_GIZMO_LIVE_LOG")
    if path:
        kw = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
        with open(path, "a") as fh:
            fh.write(json.dumps({"event": event, **kw}) + "\n")
_AUTO = {"installed": False, "sessions": set()}


def _pvs():
    import paraview.simple as pvs  # client-side only; imported lazily
    return pvs


def view_window(view):
    """(center, normal, view_up, size) covering what `view` shows at its focal plane.

    normal points from the focal point to the camera, so image right/up are
    screen right/up. size covers the larger viewport dimension (square image).
    """
    cam = view.GetActiveCamera()
    pos, focal = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
    normal = pos - focal
    dist = np.linalg.norm(normal)
    if cam.GetParallelProjection():
        height = 2.0 * cam.GetParallelScale()
    else:
        height = 2.0 * dist * np.tan(np.radians(cam.GetViewAngle()) / 2)
    w, h = (list(view.ViewSize) + [1, 1])[:2]
    aspect = (w / h) if (w > 0 and h > 0) else 1.0
    return focal, normal / dist, np.array(cam.GetViewUp()), height * max(1.0, aspect)


class CameraLink:
    """Keeps a set of Meshoid filters aimed at one view's camera."""

    def __init__(self, view, interactive_resolution=128):
        self.view = view
        self.interactive_resolution = int(interactive_resolution)
        self.filters = {}  # proxy -> full resolution
        self._last = None
        self._busy = False
        self._interacting = False
        self._observers = []
        # Keep the vtkCamera: view.GetActiveCamera() (the proxy wrapper) first pushes the
        # view's camera properties into it, which fires ModifiedEvent -> _on_camera ->
        # GetActiveCamera() ... an endless loop that froze the GUI.
        self._cam = cam = view.GetActiveCamera()
        self._observers.append((cam, cam.AddObserver("ModifiedEvent", self._on_camera)))
        # Mouse interaction start/end: the Qt GUI reports these on the view proxy
        # (vtkSMRenderViewProxy forwards them); a plain VTK interactor reports them itself.
        for src in (view.SMProxy, view.GetInteractor()):
            if src is not None:
                self._observers.append((src, src.AddObserver("StartInteractionEvent", self._on_start)))
                self._observers.append((src, src.AddObserver("EndInteractionEvent", self._on_end)))

    # -- membership
    def add(self, filt):
        res = filt.Resolution if filt.GetProperty("Resolution") is not None else 0
        self.filters[filt] = int(res[0] if hasattr(res, "__len__") else res)
        self.apply(interactive=False)

    def remove(self, filt):
        res = self.filters.pop(filt, None)
        if res and self._alive(filt):
            filt.Resolution = res

    def close(self):
        for obj, tag in self._observers:
            obj.RemoveObserver(tag)
        self._observers = []
        for f in list(self.filters):
            self.remove(f)

    @staticmethod
    def _alive(filt):
        """Still registered as a pipeline source (Delete() unregisters but leaves the Python object)."""
        try:
            from paraview import servermanager as sm
            return bool(sm.ProxyManager().SMProxyManager.IsProxyInGroup(filt.SMProxy, "sources"))
        except Exception:
            return False

    def _rep(self, filt):
        try:
            from paraview import servermanager as sm
            return sm.GetRepresentation(filt, self.view)
        except Exception:
            return None

    # -- camera handling
    def _state(self):
        cam = self._cam
        return tuple(np.round(np.r_[cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp(),
                                    cam.GetViewAngle(), cam.GetParallelScale(), cam.GetParallelProjection(),
                                    list(self.view.ViewSize)[:2]], 12))

    def _on_camera(self, *args):
        # Camera ModifiedEvent also fires for clipping-range updates during every
        # render: only react to geometric changes, and never re-enter.
        if self._busy:
            return
        self._busy = True
        try:
            st = self._state()
        finally:
            self._busy = False
        if st == self._last:
            return
        self.apply(interactive=self._interacting)

    def _on_start(self, *args):
        self._interacting = True
        debug_log("start")

    def _on_end(self, *args):
        if not self._interacting:  # reported by both sources: handle once
            return
        self._interacting = False
        debug_log("end")
        self.apply(interactive=False)  # back to full resolution
        _pvs().Render(self.view)

    def apply(self, interactive=False):
        """Aim all linked filters at the current camera."""
        self._busy = True
        try:
            self._last = self._state()
            center, normal, up, size = view_window(self.view)
            for f, full in list(self.filters.items()):
                if not self._alive(f):
                    self.filters.pop(f, None)
                    continue
                if not full:  # no image of its own (Star Properties): only the view's scale
                    f.ReferenceSize = float(size)
                    continue
                f.AutoWindow = 0
                if f.GetProperty("Normal") is None:
                    # axis-aligned 3D grid (Meshoid Grid): only centre and zoom matter, and
                    # it stays at full resolution. Rounded so that a pure rotation (same
                    # values up to float noise) doesn't re-grid.
                    s = float(f"{size:.6g}")
                    f.Center = list(np.round(np.asarray(center) / (s * 1e-6)) * (s * 1e-6))
                    f.Size = s
                    set_volume_scale(self._rep(f), f)
                    continue
                f.Center = list(center)
                f.Size = float(size)
                f.Normal = list(normal)
                f.ViewUp = list(up)
                f.Resolution = min(full, self.interactive_resolution) if interactive else full
            debug_log("aim", normal=normal, size=size, res=max(self.filters.values(), default=None),
                      interactive=interactive)
        finally:
            self._busy = False


VOLUME_UNIT_FRACTION = 0.1


def set_volume_scale(rep, filt):
    """Volume opacity is per unit world length: tie that length to the grid size, so a
    zoomed-in (re-gridded) volume looks as opaque as before instead of fading out."""
    if rep is None or rep.GetProperty("ScalarOpacityUnitDistance") is None:
        return
    size = float(filt.Size) if not int(filt.AutoWindow) else None
    if size is None:
        b = filt.GetDataInformation().GetBounds()
        size = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    if size > 0:
        rep.ScalarOpacityUnitDistance = VOLUME_UNIT_FRACTION * size


def _active_render_view():
    pvs = _pvs()
    v = pvs.GetActiveView()
    if v is None or v.GetXMLName() != "RenderView":
        v = pvs.GetActiveViewOrCreate("RenderView")
    return v


def link_for(view):
    key = id(view.SMProxy)
    if key not in _LINKS:
        _LINKS[key] = CameraLink(view)
    return _LINKS[key]


def follow_camera(filt=None, view=None, interactive_resolution=None):
    """Make a Meshoid Projection/Slice follow `view`'s camera (default: active ones)."""
    pvs = _pvs()
    filt = filt or pvs.GetActiveSource()
    view = view or _active_render_view()
    unfollow(filt)
    link = link_for(view)
    if interactive_resolution is not None:
        link.interactive_resolution = int(interactive_resolution)
    link.add(filt)
    return link


def unfollow(filt):
    for link in _LINKS.values():
        if filt in link.filters:
            link.remove(filt)


def following(filt):
    return any(filt in link.filters for link in _LINKS.values())


def stop_all():
    for link in list(_LINKS.values()):
        link.close()
    _LINKS.clear()


def _link_new_filters(*args):
    """Proxy-manager observer: link newly registered filters that have FollowCamera on."""
    pvs = _pvs()
    for (name, _), src in list(pvs.GetSources().items()):
        try:
            if src.GetProperty("FollowCamera") is not None and int(src.FollowCamera) \
                    and not following(src):
                follow_camera(src, _active_render_view())
        except Exception:
            continue


def in_gui():
    """True inside the ParaView Qt client (not pvpython/pvbatch/pvserver).

    PV_GIZMO_AUTO_FOLLOW=1/0 forces it (tests, unusual launchers).
    """
    import os
    import sys
    env = os.environ.get("PV_GIZMO_AUTO_FOLLOW")
    if env is not None:
        return env == "1"
    try:
        exe = os.path.basename(os.readlink("/proc/self/exe"))
        return "paraview" in exe and not exe.startswith(("pvpython", "pvbatch", "pvserver"))
    except OSError:
        return sys.argv[:1] == [""]  # the GUI's embedded interpreter has an empty argv


def enable_auto_follow():
    """Auto-link filters with FollowCamera=1 as they are created (GUI client only).

    Installed per session: in client-server mode the GUI starts with a builtin session
    and then connects, so a hook installed at plugin load would watch the wrong session.
    Called at plugin load and again whenever a GIZMO filter is constructed. Scripts
    (pvpython) are left alone so explicit windows are not overridden; use follow_camera()
    there. Safe to call repeatedly.
    """
    if not in_gui():
        return False
    try:
        from paraview import servermanager as sm
        if sm.vtkProcessModule.GetProcessType() != sm.vtkProcessModule.PROCESS_CLIENT:
            return False
        pxm = sm.vtkSMProxyManager.GetProxyManager().GetActiveSessionProxyManager()
        if pxm is None:
            return False
        key = pxm.GetAddressAsString("")
        if key in _AUTO["sessions"]:
            return True
        from vtkmodules.vtkCommonCore import vtkCommand
        pxm.AddObserver(vtkCommand.RegisterEvent, _link_new_filters)
        _AUTO["sessions"].add(key)
        _AUTO["installed"] = True
        return True
    except Exception:
        return False
