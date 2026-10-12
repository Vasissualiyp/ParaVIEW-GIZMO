"""Client-side conveniences for the ParaView GUI / pvpython (need paraview.simple).

    from pvgizmo import pvhelpers as ph
    reader = ph.open_snapshot("snapshot_100.hdf5", Recenter=1)
    ph.show_particles(reader)                 # Point Gaussian preset, gas coloured by density
    proj = ph.add_projection(reader)          # column density, log colour map
    ph.project_along_camera(proj)             # re-aim the projection at the current view
"""

import numpy as np
import paraview.simple as pvs


def open_snapshot(filenames, **props):
    """GIZMO reader on one file or a list (time series); runs it once so array lists are populated."""
    if isinstance(filenames, str):
        filenames = [filenames]
    r = pvs.GIZMOSnapshotReader(FileNames=list(filenames), **props)
    r.UpdatePipelineInformation()
    r.UpdatePipeline()
    return r


def camera_window(view=None):
    """(center, normal, view_up, size) of what the camera currently shows, at its focal plane."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    cam = view.GetActiveCamera()
    pos, focal = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
    normal = pos - focal  # line of sight, pointing at the viewer: image right/up = screen right/up
    if cam.GetParallelProjection():
        height = 2.0 * cam.GetParallelScale()
    else:
        height = 2.0 * np.linalg.norm(normal) * np.tan(np.radians(cam.GetViewAngle()) / 2)
    return focal, normal / np.linalg.norm(normal), np.array(cam.GetViewUp()), height


def project_along_camera(filt=None, view=None, render=True):
    """Aim a Meshoid Projection/Slice at the current camera: same centre, orientation and field of view."""
    filt = filt or pvs.GetActiveSource()
    center, normal, up, size = camera_window(view)
    filt.AutoWindow = 0
    filt.Center = list(center)
    filt.Normal = list(normal)
    filt.ViewUp = list(up)
    filt.Size = float(size)
    if render:
        pvs.Render(view)
    return filt


def show_particles(source, block="Gas", view=None, color_by="Density", opacity=0.3):
    """Point Gaussian splats with radius H/2 (VTK h), coloured by `color_by` on a log scale."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    ext = pvs.ExtractBlock(Input=source, Selectors=[f"/Root/{block}"])
    rep = pvs.Show(ext, view)
    rep.SetRepresentationType("Point Gaussian")
    rep.ShaderPreset = "Gaussian Blur"
    rep.ScaleByArray = 1
    rep.SetScaleArray = ["POINTS", "SmoothingLengthVTK"]
    rep.UseScaleFunction = 0
    rep.GaussianRadius = 1.0
    rep.Opacity = opacity
    if color_by:
        pvs.ColorBy(rep, ("POINTS", color_by))
        pvs.GetColorTransferFunction(color_by).UseLogScale = 1
        rep.RescaleTransferFunctionToDataRange(False, True)
    pvs.ResetCamera(view)
    return ext, rep


def map_range(filt, log=True, percentiles=(1.0, 99.9)):
    """Colour range from percentiles of the image values (positive ones for log maps).

    Robust for high-dynamic-range maps (collapsing clouds span many decades), and
    log maps can't include the empty (zero) pixels of a projection. The image is
    fetched to the client, which is cheap (res^2 values).
    """
    from paraview import servermanager
    from vtkmodules.util import numpy_support as ns
    img = servermanager.Fetch(filt)
    a = ns.vtk_to_numpy(img.GetPointData().GetArray(0)).astype(np.float64)
    if a.ndim > 1:
        a = np.linalg.norm(a, axis=1)
    a = a[np.isfinite(a)]
    if log:
        a = a[a > 0]
    if a.size == 0:
        return None
    lo, hi = np.percentile(a, percentiles)
    return (lo, hi) if hi > lo else (a.min(), a.max())


def show_map(filt, view=None, log=True, preset="Inferno (matplotlib)", percentiles=(1.0, 99.9)):
    """Show a Meshoid Projection/Slice image with a (log) colour map over robust percentiles."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    filt.UpdatePipeline()
    name = filt.PointData.keys()[0]
    rep = pvs.Show(filt, view)
    rep.SetRepresentationType("Slice")
    pvs.ColorBy(rep, ("POINTS", name))
    lut = pvs.GetColorTransferFunction(name)
    lut.ApplyPreset(preset, True)
    rng = map_range(filt, log, percentiles)
    if rng is not None:
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(*rng)
        lut.UseLogScale = int(bool(log))
    return rep


def add_projection(source, block="Gas", field="Masses", mode="Surface density", view=None, **props):
    """Meshoid Projection of `field` and show it; extra props are passed to the filter."""
    p = pvs.MeshoidProjection(Input=source, Block=block, Field=["POINTS", field], Mode=mode, **props)
    show_map(p, view)
    return p


def add_slice(source, block="Gas", field="Density", view=None, **props):
    s = pvs.MeshoidSlice(Input=source, Block=block, Field=["POINTS", field], **props)
    show_map(s, view)
    return s
