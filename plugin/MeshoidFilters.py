"""ParaView filters: kernel-weighted projections and slices of GIZMO particles via meshoid.

Input: the GIZMO reader's output (or any point set / composite of point sets);
the particle type is chosen by block name. Output: a 2D vtkStructuredGrid placed in
3D at the projection/slice plane (node grid spanning the window, see
pvgizmo.meshoid_ops), so it overlays the particles.
"""

import os
import sys

_ROOT = os.environ.get("PV_GIZMO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy  # noqa: E402
from vtkmodules.util import numpy_support as ns  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkImageData, vtkPointSet  # noqa: E402
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP  # noqa: E402

from pvgizmo import meshoid_ops as mo  # noqa: E402  (imports pvgizmo first: multiprocessing fix)
from pvgizmo import live as _live  # noqa: E402
from pvgizmo import vtkconv  # noqa: E402

from pvgizmo.pvbase import (  # noqa: E402,F401  (_CACHE: tests check the shared tree)
    _CACHE, _VIEW_CACHE, DES_NGB, MODE_LABELS, _arr, _bool_xml, _common_props, _INPUT_ARRAY_XML,
    _meshoid_for, _MeshoidImageFilter, _parallel_piece, _points_key, _replace_block, _resolve_field, project_field)


@smproxy.filter(name="MeshoidProjection", label="Meshoid Projection")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props  # innermost: smproxy.filter reads the properties when it runs
class MeshoidProjection(_MeshoidImageFilter):
    """Kernel-weighted projection (column density, LOS integral, averages) along Normal."""

    def __init__(self):
        _MeshoidImageFilter.__init__(self)
        self._mode = 0
        self._depth = 0.0
        self._smooth = 1.0
        self._conservative = False

    @smproperty.xml(_INPUT_ARRAY_XML.format(default="Masses"))
    def SetInputArrayToProcess(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    @smproperty.xml("""
<IntVectorProperty name="Mode" command="SetMode" number_of_elements="1" default_values="0">
  <EnumerationDomain name="enum">""" + "".join(
        f'<Entry value="{i}" text="{t}"/>' for i, t in enumerate(MODE_LABELS)) + """</EnumerationDomain>
  <Documentation>Surface density: integral of the density of a conserved per-particle Field (Masses
  -> column density). Line-of-sight integral: integral of an intensive Field along the sightline.
  Mass-weighted average: integral(rho f)/integral(rho). Projected average: number-weighted.</Documentation>
</IntVectorProperty>""")
    def SetMode(self, m):
        self._set("_mode", int(m))

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 1,
        "GUI: keep this projection aimed at the render view's camera (direction, centre, zoom), "
        "updating as you rotate/zoom/pan. Overrides Normal/ViewUp/Center/Size/AutoWindow."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)  # acted on by the client (pvgizmo.live); nothing to do here

    @smproperty.doublevector(name="Depth", default_values=0.0)
    def SetDepth(self, d):
        self._set("_depth", float(d))

    @smproperty.doublevector(name="SmoothingFactor", default_values=1.0)
    def SetSmoothingFactor(self, s):
        self._set("_smooth", float(s))

    @smproperty.xml(_bool_xml("Conservative", "SetConservative", 0,
                              "Mass-conserving deposition (Surface density mode only)."))
    def SetConservative(self, v):
        self._set("_conservative", bool(v))

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
        if pd is None:
            return 1  # empty piece
        name = _resolve_field(pd, self._field_name(), "Masses")
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        f = _arr(pd, name)
        if f is None:
            raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
        m = _arr(pd, "Masses")
        h = self._kernel_radius(pd, pos, m)
        mode = mo.MODES[self._mode]
        if mode == "MassWeightedAverage" and m is None:
            raise RuntimeError("Mass-weighted average needs a Masses array")
        center, size = self._window(pos, pd)
        self.UpdateProgress(0.1)
        if mode == "SurfaceDensity" and self._conservative:
            img = mo.project(mode, pos, h, center, size, self._res, normal=self._normal, view_up=self._up,
                             field=f, masses=m, smooth_fac=self._smooth, depth=self._depth, conservative=True)
        else:
            img = project_field(pd, name, mode, pos, h, center, size, self._res, self._normal, self._up,
                                self._smooth, self._depth)
        self._emit(outInfoVec, img, f"{mode}_{name}", center, size)
        _live.debug_log("exec", filter="projection", normal=list(self._normal), res=self._res)
        return 1


@smproxy.filter(name="MeshoidSlice", label="Meshoid Slice")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props  # innermost: smproxy.filter reads the properties when it runs
class MeshoidSlice(_MeshoidImageFilter):
    """Field reconstructed on a plane (normal = Normal, through Center) from the nearest particle."""

    def __init__(self):
        _MeshoidImageFilter.__init__(self)
        self._order = 0
        self._limiter = False

    @smproperty.xml(_INPUT_ARRAY_XML.format(default="Density"))
    def SetInputArrayToProcess(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    @smproperty.xml("""
<IntVectorProperty name="Order" command="SetOrder" number_of_elements="1" default_values="0">
  <EnumerationDomain name="enum"><Entry value="0" text="0 (nearest particle)"/>
    <Entry value="1" text="1 (linear)"/><Entry value="2" text="2 (quadratic)"/></EnumerationDomain>
</IntVectorProperty>""")
    def SetOrder(self, o):
        self._set("_order", int(o))

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 0,
        "GUI: keep the slice through the focal point, facing the camera, as the view moves."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)

    @smproperty.xml(_bool_xml("SlopeLimiter", "SetSlopeLimiter", 0,
                              "Barth-Jespersen limiter on the gradient (as in GIZMO), order >= 1."))
    def SetSlopeLimiter(self, v):
        self._set("_limiter", bool(v))

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
        if pd is None:
            return 1  # empty piece
        name = _resolve_field(pd, self._field_name(), "Density")
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        f = _arr(pd, name)
        if f is None:
            raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
        M = _meshoid_for(pd, pos)
        center, size = self._window(pos)
        self.UpdateProgress(0.1)
        img = mo.slice_field(M, f, center, size, self._res, self._normal, self._up,
                             order=self._order, slope_limiter=self._limiter)
        self._emit(outInfoVec, img, f"Slice_{name}", center, size)
        return 1


GRID_MODES = ("Deposit (density of a conserved field)", "Interpolate (intensive field)")


@smproxy.filter(name="MeshoidGrid", label="Meshoid Grid")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props(["SetBlock", "SetAutoWindow", "SetCenter", "SetSize", "SetResolution"])
class MeshoidGrid(_MeshoidImageFilter):
    """Particles -> axis-aligned 3D vtkImageData (volume rendering, contours, streamlines).

    Deposit: density of a conserved per-particle field (Masses -> mass density),
    exactly conservative. Interpolate: intensive field reconstructed at the nodes
    (order 0-2); vector fields (Velocities, MagneticField) give vector grids for
    Stream Tracer / Glyph. Nodes coincide with the Projection/Slice nodes.
    """

    def __init__(self):
        _MeshoidImageFilter.__init__(self, outputType="vtkImageData")  # axis-aligned: no direction matrix
        self._res = 64
        self._mode = 0
        self._order = 0
        self._limiter = False
        self._log10 = True
        self._decades = 4.0
        self._scale_range = (0.0, 0.0)
        self._follow = False

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 0,
        "GUI: keep the grid centred on the camera's focal point with the visible width as its "
        "size, re-gridding as you zoom/pan (rotation needs no re-gridding). Overrides "
        "Center/Size/AutoWindow."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)  # acted on by the client (pvgizmo.live)

    @smproperty.doublevector(name="DynamicRange", default_values=4.0)
    def SetDynamicRange(self, d):
        """Decades covered by scaled_log10_[array]."""
        self._set("_decades", max(float(d), 1e-3))

    @smproperty.xml("""
<DoubleVectorProperty name="ScaleRange" command="SetScaleRange" number_of_elements="2" default_values="0 0">
  <Documentation>Fixed log10 levels for scaled_log10_[array]: 0 at the first, 1 at the second
  (e.g. log10 density in code units). Same opacity for the same density in every frame and at
  every zoom: use it for movies. 0 0 = automatic (DynamicRange decades below the densest voxels
  of the current window).</Documentation>
</DoubleVectorProperty>""")
    def SetScaleRange(self, lo, hi):
        self._set("_scale_range", (float(lo), float(hi)))

    @smproperty.xml(_bool_xml(
        "AddLog10", "SetAddLog10", 1,
        "Also output log10_[array] (empty voxels set one decade below the smallest positive value) and "
        "scaled_log10_[array]: 0 at DynamicRange decades below the densest voxels (99.9th percentile), "
        "1 at them. Use log10 for contours and scaled_log10 for volume rendering: its fixed 0-1 range "
        "keeps one opacity curve valid at every zoom and timestep. (ParaView's volume mapper ignores "
        "log-scaled colour maps.)"))
    def SetAddLog10(self, v):
        self._set("_log10", bool(v))

    @smproperty.xml(_INPUT_ARRAY_XML.format(default="Masses"))
    def SetInputArrayToProcess(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    @smproperty.xml("""
<IntVectorProperty name="Mode" command="SetMode" number_of_elements="1" default_values="0">
  <EnumerationDomain name="enum">""" + "".join(
        f'<Entry value="{i}" text="{t}"/>' for i, t in enumerate(GRID_MODES)) + """</EnumerationDomain>
</IntVectorProperty>""")
    def SetMode(self, m):
        self._set("_mode", int(m))

    @smproperty.xml("""
<IntVectorProperty name="Order" command="SetOrder" number_of_elements="1" default_values="0">
  <EnumerationDomain name="enum"><Entry value="0" text="0 (nearest particle)"/>
    <Entry value="1" text="1 (linear)"/><Entry value="2" text="2 (quadratic)"/></EnumerationDomain>
  <Documentation>Reconstruction order (Interpolate mode).</Documentation>
</IntVectorProperty>""")
    def SetOrder(self, o):
        self._set("_order", int(o))

    @smproperty.xml(_bool_xml("SlopeLimiter", "SetSlopeLimiter", 0,
                              "Barth-Jespersen limiter on the gradient (Interpolate, order >= 1)."))
    def SetSlopeLimiter(self, v):
        self._set("_limiter", bool(v))

    def _levels(self):
        lo, hi = self._scale_range
        return (lo, hi) if hi > lo else None

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        info = outInfoVec.GetInformationObject(0)
        r = self._res - 1
        info.Set(SDDP.WHOLE_EXTENT(), 0, r, 0, r, 0, r)
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
        if pd is None:
            return 1  # empty piece
        name = _resolve_field(pd, self._field_name(), "Masses")
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        f = _arr(pd, name)
        if f is None:
            raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
        center, size = self._window(pos)
        self.UpdateProgress(0.1)
        if self._mode == 0:
            h = self._kernel_radius(pd, pos, _arr(pd, "Masses"))
            vals, label = mo.grid3d_deposit(pos, h, f, center, size, self._res), f"GridDensity_{name}"
        else:
            vals = mo.grid3d_interp(_meshoid_for(pd, pos), f, center, size, self._res,
                                    order=self._order, slope_limiter=self._limiter)
            label = f"Interp_{name}"
        origin, spacing = mo.grid3d_geometry(center, size, self._res)
        img = vtkconv.image_from_array(vals, origin, spacing, name=label)
        if self._log10 and vals.ndim == 3:
            lg = mo.safe_log10(vals)
            img.GetPointData().AddArray(vtkconv.to_vtk_array(np.ravel(lg, order="F"), f"log10_{label}"))
            img.GetPointData().AddArray(vtkconv.to_vtk_array(
                np.ravel(mo.scaled_log10(vals, self._decades, levels=self._levels()), order="F"), f"scaled_log10_{label}"))
        out = vtkImageData.GetData(outInfoVec, 0)
        out.ShallowCopy(img)
        vtkconv.add_field_data(out, {"WindowCenter": center, "WindowSize": size})
        return 1



@smproxy.filter(name="MeshoidDerivedFields", label="Meshoid Derived Fields")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props(["SetBlock"])
class MeshoidDerivedFields(VTKPythonAlgorithmBase):
    """Adds meshoid-derived point arrays to one particle type; everything else passes through.

    Gradient (least-squares, order 1 or 2), divergence & curl (3-vectors), kernel
    average / standard deviation of Field, and meshoid's own kernel radius, density
    and volume (des_ngb = 32; named Meshoid* to keep them apart from GIZMO's).
    """

    _FLAGS = (("ComputeGradient", "Gradient", 1), ("ComputeDivCurl", "DivCurl", 0),
              ("ComputeKernelAverage", "KernelAverage", 0), ("ComputeKernelStd", "KernelStd", 0),
              ("ComputeKernelRadius", "KernelRadius", 0), ("ComputeDensity", "Density", 0),
              ("ComputeVolume", "Volume", 0))

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkDataObject")
        self._block = "Gas"
        self._order = 1
        self._which = {key: bool(d) for _, key, d in self._FLAGS}

    def FillInputPortInformation(self, port, info):
        info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataObject")
        return 1

    def RequestDataObject(self, request, inInfoVec, outInfoVec):
        inp = vtkDataObject.GetData(inInfoVec[0], 0)
        out = vtkDataObject.GetData(outInfoVec, 0)
        if inp is not None and (out is None or out.GetClassName() != inp.GetClassName()):
            outInfoVec.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), inp.NewInstance())
        return 1

    def _set(self, attr, value):
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            self.Modified()

    def SetBlock(self, name):
        self._set("_block", str(name))

    def _flag(self, key, v):
        if self._which[key] != bool(v):
            self._which[key] = bool(v)
            self.Modified()

    @smproperty.xml(_INPUT_ARRAY_XML.format(default="Density"))
    def SetInputArrayToProcess(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    @smproperty.xml("""
<IntVectorProperty name="GradientOrder" command="SetGradientOrder" number_of_elements="1" default_values="1">
  <EnumerationDomain name="enum"><Entry value="1" text="1"/><Entry value="2" text="2"/></EnumerationDomain>
  <Documentation>Order of the least-squares fit used for gradients (2 is more accurate, slower).</Documentation>
</IntVectorProperty>""")
    def SetGradientOrder(self, o):
        self._set("_order", int(o))

    @smproperty.xml(_bool_xml("ComputeGradient", "SetComputeGradient", 1, "grad_[Field]: 3 comps, 9 for vectors ([i*3+k] = d f_i/d x_k)."))
    def SetComputeGradient(self, v):
        self._flag("Gradient", v)

    @smproperty.xml(_bool_xml("ComputeDivCurl", "SetComputeDivCurl", 0, "div_[Field], curl_[Field] for 3-vector fields."))
    def SetComputeDivCurl(self, v):
        self._flag("DivCurl", v)

    @smproperty.xml(_bool_xml("ComputeKernelAverage", "SetComputeKernelAverage", 0, "Kernel-weighted average of Field over neighbours."))
    def SetComputeKernelAverage(self, v):
        self._flag("KernelAverage", v)

    @smproperty.xml(_bool_xml("ComputeKernelStd", "SetComputeKernelStd", 0, "Standard deviation of Field over neighbours."))
    def SetComputeKernelStd(self, v):
        self._flag("KernelStd", v)

    @smproperty.xml(_bool_xml("ComputeKernelRadius", "SetComputeKernelRadius", 0, "MeshoidKernelRadius (meshoid's H, des_ngb=32)."))
    def SetComputeKernelRadius(self, v):
        self._flag("KernelRadius", v)

    @smproperty.xml(_bool_xml("ComputeDensity", "SetComputeDensity", 0, "MeshoidDensity (e.g. for stars/DM)."))
    def SetComputeDensity(self, v):
        self._flag("Density", v)

    @smproperty.xml(_bool_xml("ComputeVolume", "SetComputeVolume", 0, "MeshoidVolume."))
    def SetComputeVolume(self, v):
        self._flag("Volume", v)

    def RequestData(self, request, inInfoVec, outInfoVec):
        inp = vtkDataObject.GetData(inInfoVec[0], 0)
        out = vtkDataObject.GetData(outInfoVec, 0)
        out.ShallowCopy(inp)
        selected = self.GetInputArrayInformation(0).Get(vtkDataObject.FIELD_NAME())
        which = [k for k, on in self._which.items() if on]
        if not which:
            return 1

        def add(pd):
            name = _resolve_field(pd, selected, "Density")
            pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
            f = _arr(pd, name)
            needs_f = {"Gradient", "DivCurl", "KernelAverage", "KernelStd"} & set(which)
            if needs_f and f is None:
                raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
            sel = [w for w in which if not (w == "DivCurl" and (f is None or f.ndim != 2 or f.shape[1] != 3))]
            arrays = mo.derived_fields(_meshoid_for(pd, pos), f, name, which=sel, gradient_order=self._order)
            for k, a in arrays.items():
                pd.GetPointData().AddArray(vtkconv.to_vtk_array(a, k))
            return pd

        if isinstance(out, vtkPointSet):
            if out.GetNumberOfPoints():
                add(out)
        elif not _replace_block(out, self._block, add) and not _parallel_piece(inInfoVec):
            raise RuntimeError(f"no particles in block '{self._block}'")
        return 1


# Client-server GUI: the filter constructor runs on the server, so the FollowCamera
# auto-linker must also be installed here, when the client imports the plugin (after
# connecting, so a session exists). In the builtin GUI there is no session yet at
# import; the constructor-time call above covers that case. No-op outside the GUI.
try:
    _live.enable_auto_follow()
except Exception:
    pass

