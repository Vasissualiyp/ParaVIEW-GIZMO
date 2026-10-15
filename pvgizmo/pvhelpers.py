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


def map_range(filt, log=True, percentiles=(1.0, 99.9), name=None):
    """Colour range from percentiles of the image values (positive ones for log maps).

    Robust for high-dynamic-range maps (collapsing clouds span many decades), and
    log maps can't include the empty (zero) pixels of a projection. The image is
    fetched to the client, which is cheap (res^2 values).
    """
    from paraview import servermanager
    from vtkmodules.util import numpy_support as ns
    img = servermanager.Fetch(filt)
    arr = img.GetPointData().GetArray(name) if name else img.GetPointData().GetArray(0)
    a = ns.vtk_to_numpy(arr).astype(np.float64)
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
    rep.SetRepresentationType("Surface")
    pvs.ColorBy(rep, ("POINTS", name))
    lut = pvs.GetColorTransferFunction(name)
    lut.ApplyPreset(preset, True)
    rng = map_range(filt, log, percentiles, name)
    if rng is not None:
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(*rng)
        lut.UseLogScale = int(bool(log))
    return rep


def show_volume(grid, view=None, preset="Inferno (matplotlib)", color_pct=(50.0, 99.99),
                opacity_pct=(90.0, 99.99), max_opacity=0.6):
    """Volume-render a Meshoid Grid using its log10_<array> (linear map over log values).

    ParaView's volume mapper ignores log-scaled colour maps, hence the log10 array
    (Meshoid Grid's AddLog10, on by default). Opacity starts higher (opacity_pct)
    than the colour range (color_pct), so visible voxels are coloured, not black.
    """
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    grid.UpdatePipeline()
    names = grid.PointData.keys()
    name = next((n for n in names if n.startswith("log10_")), names[0])
    rep = pvs.Show(grid, view)
    rep.SetRepresentationType("Volume")
    pvs.ColorBy(rep, ("POINTS", name))
    lut = pvs.GetColorTransferFunction(name)
    lut.ApplyPreset(preset, True)
    c = map_range(grid, False, color_pct, name)
    o = map_range(grid, False, opacity_pct, name)
    if c is None or o is None:
        return rep
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(*c)
    otf = pvs.GetOpacityTransferFunction(name)
    otf.Points = [c[0], 0.0, 0.5, 0.0, o[0], 0.0, 0.5, 0.0, c[1], max_opacity, 0.5, 0.0]
    return rep


def add_projection(source, block="Gas", field="Masses", mode="Surface density", view=None,
                   follow=False, **props):
    """Meshoid Projection of `field` and show it; extra props are passed to the filter.

    follow=True keeps it aimed at the view's camera (pvgizmo.live); in the GUI this
    happens automatically for FollowCamera=1 (the default).
    """
    p = pvs.MeshoidProjection(Input=source, Block=block, Field=["POINTS", field], Mode=mode, **props)
    if follow:
        from pvgizmo import live
        live.follow_camera(p, view)
    show_map(p, view)
    return p


def add_slice(source, block="Gas", field="Density", view=None, follow=False, **props):
    s = pvs.MeshoidSlice(Input=source, Block=block, Field=["POINTS", field], **props)
    if follow:
        from pvgizmo import live
        live.follow_camera(s, view)
    show_map(s, view)
    return s


# --------------------------------------------------------------------------- space look (Phase 6)
# Point Gaussian fragment code: white-hot core, soft halo and faint diffraction spikes. With
# Emissive on, the mapper blends additively, so overlapping stars add up and glow.
STAR_SHADER = """//VTK::Color::Impl
  float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
  if (r2 > 9.0) { discard; }
  float core = exp(-8.0 * r2);
  float halo = 0.25 * exp(-1.2 * r2);
  float spike = 0.12 * (exp(-40.0 * offsetVCVSOutput.x * offsetVCVSOutput.x)
                      + exp(-40.0 * offsetVCVSOutput.y * offsetVCVSOutput.y)) * exp(-0.6 * r2);
  diffuseColor = mix(diffuseColor, vec3(1.0), 0.7 * clamp(core, 0.0, 1.0));
  ambientColor = diffuseColor;
  opacity = opacity * clamp(core + halo + spike, 0.0, 1.0);
"""
# appended when stars must draw over opaque maps (which sit at the focal plane): keep the
# depth order among stars but move them all in front of everything else
STAR_ON_TOP = "  gl_FragDepth = 1e-3 * gl_FragCoord.z;\n"


def space_view(view=None):
    """Black background, no orientation axes: the canvas for glowing gas and stars."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = [0.0, 0.0, 0.0]
    view.OrientationAxesVisibility = 0
    return view


def show_rgb(filt, view=None):
    """Show an RGB map (Starlight Image, Meshoid Composite) with its colours used directly."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    rep = pvs.Show(filt, view)
    rep.SetRepresentationType("Surface")
    pvs.ColorBy(rep, ("POINTS", "RGB"))
    rep.MapScalars = 0
    rep.Ambient, rep.Diffuse = 1.0, 0.0  # unlit: the image's colours, not shaded ones
    rep.SetScalarBarVisibility(view, False)
    return rep


def show_space_map(filt, view=None, preset="Magma (matplotlib)", percentiles=(1.0, 99.9)):
    """A Meshoid Projection as glowing gas: log colour map on black (vizmo's default look)."""
    view = space_view(view)
    rep = show_map(filt, view, log=True, preset=preset, percentiles=percentiles)
    rep.Ambient, rep.Diffuse = 1.0, 0.0
    rep.SetScalarBarVisibility(view, False)
    return rep


def show_stars(source, block="Stars", view=None, size_scale=2.0, opacity=1.0, on_top=True, **props):
    """Glowing stars: GIZMO Star Properties on `block`, drawn as additive Point Gaussian splats.

    Colour = StarRGB (band ratios: young blue, old red), size grows with log L. on_top: draw
    over maps (images are opaque planes through the focal point). Returns (filter, rep).
    """
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    sp = pvs.GIZMOStarProperties(Input=source, Block=block, SizeScale=size_scale, **props)
    ext = pvs.ExtractBlock(Input=sp, Selectors=[f"/Root/{block}"])
    rep = pvs.Show(ext, view)
    rep.SetRepresentationType("Point Gaussian")
    rep.ShaderPreset = "Custom"
    rep.CustomShader = STAR_SHADER + (STAR_ON_TOP if on_top else "")
    rep.Emissive = 1
    rep.ScaleByArray = 1
    rep.SetScaleArray = ["POINTS", "StarSize"]
    rep.UseScaleFunction = 0
    rep.GaussianRadius = 1.0
    rep.Opacity = opacity
    pvs.ColorBy(rep, ("POINTS", "StarRGB"))
    rep.MapScalars = 0
    rep.Ambient, rep.Diffuse = 1.0, 0.0
    rep.SetScalarBarVisibility(view, False)
    return sp, rep


def add_starlight(source, view=None, follow=False, **props):
    """GIZMO Starlight Image (FIRE-style three-band stars with dust) on a black view."""
    view = space_view(view)
    s = pvs.GIZMOStarlightImage(Input=source, **props)
    if follow:
        from pvgizmo import live
        live.follow_camera(s, view)
    show_rgb(s, view)
    return s


def add_composite(source, view=None, follow=False, **props):
    """Meshoid Composite (brightness = column density, colour = e.g. temperature) on a black view."""
    view = space_view(view)
    if "Field2" not in props:  # the array domain would fall back to the first array (Density)
        source.UpdatePipeline()
        keys = source.PointData.keys()
        f2 = next((k for k in ("Temperature", "InternalEnergy") if k in keys), None)
        if f2:
            props["Field2"] = ["POINTS", f2]
    c = pvs.MeshoidComposite(Input=source, **props)
    if follow:
        from pvgizmo import live
        live.follow_camera(c, view)
    show_rgb(c, view)
    return c


def block_names(source):
    """Names of the top-level blocks of a composite source (e.g. ['Gas', 'Stars']), from its
    data information (nothing is fetched to the client)."""
    source.UpdatePipeline()
    h = source.GetDataInformation().DataInformation.GetHierarchy()
    if h is None:
        return []
    return [h.GetNodeName(n) for n in h.GetChildNodes(h.GetRootNode())]


def space_look(source, view=None):
    """One-call space look for a GIZMO reader: starlight image if there are star particles,
    else a column-density x temperature composite; glowing stars for STARFORGE sinks."""
    view = space_view(view)
    names = block_names(source)
    pvs.Hide(source, view)
    made = {}
    if "Stars" in names:
        made["starlight"] = add_starlight(source, view)
    elif "Gas" in names:
        made["composite"] = add_composite(source, view)
    if "Type5" in names:
        ref = 0.0
        for img in made.values():  # size the stars relative to the image they overlay
            img.UpdatePipeline()
            ref = img.FieldData["WindowSize"].GetRange()[0]
        made["stars"] = show_stars(source, block="Type5", view=view, ReferenceSize=ref)[0]
    pvs.Render(view)
    return made


# ---------------------------------------------------------------- star-centred movies
def center_on_star(reader, particle_id=0):
    """Recenter the GIZMO reader on the most massive star of every snapshot (or on the
    star with ParticleIDs == particle_id), so the star sits at the origin in each frame."""
    reader.Recenter = "Most massive star"
    reader.CenterParticleID = int(particle_id)
    reader.UpdatePipeline()
    return reader


def star_camera(view=None, size=None, parallel=True):
    """Aim the view at the origin (the star, with center_on_star), keeping the direction.

    Focal point and centre of rotation go to the origin, so rotating orbits the star and
    zooming closes in on it instead of on the middle of the whole box. Parallel projection
    makes zoom a pure change of scale: no flying past the region, no near-plane clipping.
    size: visible width (default: the current one).
    """
    from pvgizmo import live
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    _, normal, up, cur = live.view_window(view)
    size = float(size or cur)
    w, h = (list(view.ViewSize) + [1, 1])[:2]
    aspect = max(1.0, w / h) if w > 0 and h > 0 else 1.0
    height = size / aspect
    view.CameraParallelProjection = int(bool(parallel))
    dist = 0.5 * height / np.tan(np.radians(view.CameraViewAngle) / 2)
    if parallel:
        dist = max(dist, 10 * size)  # far enough that the clipping range covers the region
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CenterOfRotation = [0.0, 0.0, 0.0]
    view.CameraPosition = list(np.asarray(normal) * dist)
    view.CameraViewUp = list(up)
    view.CameraParallelScale = 0.5 * height
    view.CenterAxesVisibility = 0
    try:  # pvpython resets the camera on the first Render: that would undo all this
        pvs._DisableFirstRenderCameraReset()
    except AttributeError:
        pass
    pvs.Render(view)
    return view


def zoom(factor, view=None):
    """Zoom in by `factor` (> 1) or out (< 1) about the focal point."""
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    cam = view.GetActiveCamera()
    if view.CameraParallelProjection:
        cam.SetParallelScale(cam.GetParallelScale() / factor)
    else:
        cam.Dolly(factor)
    pvs.Render(view)


def add_dense_volume(source, view=None, block="Gas", field="Masses", resolution=96, decades=4.0,
                     preset="Inferno (matplotlib)", max_opacity=0.8, opacity_start=0.25, follow=True):
    """Volume render of the gas where dense gas is opaque and diffuse gas transparent.

    A Meshoid Grid of `field` density (mass density by default) that follows the camera
    (centred on the focal point, as wide as the view: zooming re-grids the visible region),
    rendered from scaled_log10: 0 at `decades` below the densest voxels, 1 at them. The
    opacity is 0 up to `opacity_start` and rises to `max_opacity` at the densest gas, on
    that fixed 0-1 scale, so it holds at every zoom and timestep. Edit the curve in the
    Color Map Editor. resolution 96 keeps the grid under ParaView's 20 MB remote-render
    threshold, so a remote grid is still volume-rendered on the local GPU.
    """
    from pvgizmo import live
    view = view or pvs.GetActiveViewOrCreate("RenderView")
    g = pvs.MeshoidGrid(Input=source, Block=block, Field=["POINTS", field], Resolution=int(resolution),
                        DynamicRange=float(decades), FollowCamera=int(bool(follow)))
    if follow:
        live.follow_camera(g, view)
    g.UpdatePipeline()
    name = next(n for n in g.PointData.keys() if n.startswith("scaled_log10_"))
    pvs.Hide(source, view)  # the raw particles would cover the volume
    rep = pvs.Show(g, view)
    rep.SetRepresentationType("Volume")
    pvs.ColorBy(rep, ("POINTS", name))
    lut = pvs.GetColorTransferFunction(name)
    lut.ApplyPreset(preset, True)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(0.0, 1.0)
    otf = pvs.GetOpacityTransferFunction(name)
    otf.Points = [0.0, 0.0, 0.5, 0.0, opacity_start, 0.0, 0.5, 0.0, 1.0, max_opacity, 0.5, 0.0]
    otf.RescaleTransferFunction(0.0, 1.0)
    live.set_volume_scale(rep, g)
    pvs.Render(view)
    return g
