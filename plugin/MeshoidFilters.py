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
from vtkmodules.vtkCommonDataModel import (  # noqa: E402
    vtkCompositeDataSet, vtkDataObject, vtkImageData, vtkMultiBlockDataSet, vtkPartitionedDataSet,
    vtkPartitionedDataSetCollection, vtkPointSet, vtkStructuredGrid)
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP  # noqa: E402

from pvgizmo import meshoid_ops as mo  # noqa: E402  (imports pvgizmo first: multiprocessing fix)
from pvgizmo import live as _live  # noqa: E402
from pvgizmo import vtkconv  # noqa: E402

DES_NGB = 32  # meshoid / GIZMO default neighbour number, for volumes when Density is missing
_CACHE = mo.MeshoidCache(maxsize=2)  # shared by all filter instances (KD-trees are big)


class _LRU:
    """Small LRU for view-independent per-dataset arrays (projection channels, windows)."""

    def __init__(self, maxsize=4):
        from collections import OrderedDict
        self.maxsize, self._d = maxsize, OrderedDict()

    def __contains__(self, key):
        return key in self._d

    def get(self, key):
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


_VIEW_CACHE = _LRU(maxsize=4)  # camera moves reuse these: only the deposit runs per frame

MODE_LABELS = ("Surface density", "Line-of-sight integral", "Mass-weighted average", "Projected average")

_INPUT_ARRAY_XML = """
<StringVectorProperty name="Field" label="Field" command="SetInputArrayToProcess"
                      number_of_elements="5" element_types="0 0 0 0 2"
                      default_values_delimiter=";" default_values="0;0;0;0;{default}">
  <ArrayListDomain name="array_list" input_domain_name="inputs_array">
    <RequiredProperties><Property name="Input" function="Input"/></RequiredProperties>
  </ArrayListDomain>
  <Documentation>Point array to project / slice. Vectors give one output component each.</Documentation>
</StringVectorProperty>"""


def _bool_xml(name, command, default, doc=""):
    return f"""
<IntVectorProperty name="{name}" command="{command}" number_of_elements="1" default_values="{default}">
  <BooleanDomain name="bool"/><Documentation>{doc}</Documentation>
</IntVectorProperty>"""


def find_block(obj, name):
    """The point set named `name` in a composite input (or `obj` itself if it is a point set)."""
    if isinstance(obj, vtkPointSet):
        return obj
    if isinstance(obj, vtkPartitionedDataSetCollection):
        for i in range(obj.GetNumberOfPartitionedDataSets()):
            md = obj.GetMetaData(i) if obj.HasMetaData(i) else None
            if md is not None and md.Get(vtkCompositeDataSet.NAME()) == name:
                return _first_pointset(obj.GetPartitionedDataSet(i))
        if obj.GetNumberOfPartitionedDataSets() == 1:  # e.g. after ExtractBlock
            return _first_pointset(obj.GetPartitionedDataSet(0))
        return None
    if isinstance(obj, vtkMultiBlockDataSet):
        for i in range(obj.GetNumberOfBlocks()):
            md = obj.GetMetaData(i) if obj.HasMetaData(i) else None
            if md is not None and md.Get(vtkCompositeDataSet.NAME()) == name:
                return _first_pointset(obj.GetBlock(i))
        return None
    if isinstance(obj, vtkPartitionedDataSet):
        return _first_pointset(obj)
    return None


def _first_pointset(obj):
    if isinstance(obj, vtkPointSet):
        return obj
    if isinstance(obj, vtkPartitionedDataSet):
        for k in range(obj.GetNumberOfPartitions()):
            p = obj.GetPartition(k)
            if isinstance(p, vtkPointSet) and p.GetNumberOfPoints():
                return p
    if isinstance(obj, vtkMultiBlockDataSet) and obj.GetNumberOfBlocks():
        return _first_pointset(obj.GetBlock(0))
    return None


def _points_key(pd):
    """Identity of a point set's coordinates: VTK address + modification time."""
    pts = pd.GetPoints().GetData()
    return (pts.GetAddressAsString("vtkDataArray"), pd.GetPoints().GetMTime(), pts.GetNumberOfTuples())


def _meshoid_for(pd, pos):
    """The cached Meshoid (KD-tree, neighbours, gradient weights) for this point set.

    One instance per dataset is shared by Slice, Grid (interpolate) and Derived Fields.
    """
    return _CACHE.get(_points_key(pd), pos, _arr(pd, "Masses"), _arr(pd, "SmoothingLength"))


def _resolve_field(pd, name, default):
    """The selected array name, or `default` if the selection is unset/invalid.

    A filter created before its input has run gets a placeholder ('0') from the array
    domain (scripts; the GUI always runs the input first): fall back instead of failing.
    """
    if name and pd.GetPointData().GetArray(name) is not None:
        return name
    return default


def _arr(pd, name):
    a = pd.GetPointData().GetArray(name)
    return None if a is None else ns.vtk_to_numpy(a)


class _MeshoidImageFilter(VTKPythonAlgorithmBase):
    """Shared plumbing: input block lookup, window, image output."""

    def __init__(self, outputType="vtkStructuredGrid"):
        # 2D maps are vtkStructuredGrid with explicit points, not vtkImageData + direction
        # matrix: ParaView's client-server delivery drops the direction matrix.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType=outputType)
        self._block = "Gas"
        self._normal = [0.0, 0.0, 1.0]
        self._up = [0.0, 1.0, 0.0]
        self._auto = True
        self._center = [0.0, 0.0, 0.0]
        self._size = 1.0
        self._res = 256
        # GUI (builtin session): install the FollowCamera auto-linker on first use. At plugin
        # import there is no session yet; here there is, and the proxy is not registered yet,
        # so the registration observer still catches this very filter.
        try:
            _live.enable_auto_follow()
        except Exception:
            pass

    def FillInputPortInformation(self, port, info):
        info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataObject")
        return 1

    def _set(self, attr, value):
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            self.Modified()

    # common properties (subclasses add decorators by re-declaring thin wrappers)
    def SetBlock(self, name):
        self._set("_block", str(name))

    def SetNormal(self, x, y, z):
        self._set("_normal", [x, y, z])

    def SetViewUp(self, x, y, z):
        self._set("_up", [x, y, z])

    def SetAutoWindow(self, v):
        self._set("_auto", bool(v))

    def SetCenter(self, x, y, z):
        self._set("_center", [x, y, z])

    def SetSize(self, s):
        self._set("_size", float(s))

    def SetResolution(self, n):
        self._set("_res", max(2, int(n)))

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        info = outInfoVec.GetInformationObject(0)
        info.Set(SDDP.WHOLE_EXTENT(), 0, self._res - 1, 0, self._res - 1, 0, 0)
        return 1

    def _field_name(self):
        return self.GetInputArrayInformation(0).Get(vtkDataObject.FIELD_NAME())

    def _particles(self, inInfoVec):
        pd = find_block(vtkDataObject.GetData(inInfoVec[0], 0), self._block)
        if pd is None or pd.GetNumberOfPoints() == 0:
            raise RuntimeError(f"no particles in block '{self._block}'")
        return pd

    def _window(self, pos, pd=None):
        if self._auto:
            if pd is None:
                return mo.auto_extent(pos)
            key = ("auto_extent",) + _points_key(pd)  # median/percentile: O(N log N), cache it
            if key not in _VIEW_CACHE:
                _VIEW_CACHE.put(key, mo.auto_extent(pos))
            return _VIEW_CACHE.get(key)
        return np.array(self._center, dtype=np.float64), self._size

    def _kernel_radius(self, pd, pos, masses):
        h = _arr(pd, "SmoothingLength")
        if h is not None:
            return h
        return _meshoid_for(pd, pos).SmoothingLength()  # stars/DM: meshoid kernel radius

    def _emit(self, outInfoVec, values, name, center, size):
        pts = mo.slice_points(center, size, self._res, self._normal, self._up).reshape(self._res, self._res, 3)
        grid = vtkconv.structured_grid_from_array(values, pts, name=name)
        out = vtkStructuredGrid.GetData(outInfoVec, 0)
        out.ShallowCopy(grid)
        vtkconv.add_field_data(out, {"WindowCenter": center, "WindowSize": size,
                                     "Normal": self._normal, "ViewUp": self._up})


def _common_props(cls, names=None):
    """Attach the shared properties (decorators must be applied per proxy class)."""
    if isinstance(cls, (list, tuple)):  # used as @_common_props(["SetBlock", ...])
        return lambda c: _common_props(c, cls)
    deco = [
        ("SetBlock", smproperty.stringvector(name="Block", default_values="Gas", number_of_elements=1)),
        ("SetNormal", smproperty.doublevector(name="Normal", default_values=[0, 0, 1], number_of_elements=3)),
        ("SetViewUp", smproperty.doublevector(name="ViewUp", default_values=[0, 1, 0], number_of_elements=3)),
        ("SetAutoWindow", smproperty.xml(_bool_xml(
            "AutoWindow", "SetAutoWindow", 1,
            "Centre on the particles' median, size = 2 x 90th-percentile radius (meshoid's default). "
            "Off: use Center and Size."))),
        ("SetCenter", smproperty.doublevector(name="Center", default_values=[0, 0, 0], number_of_elements=3)),
        ("SetSize", smproperty.doublevector(name="Size", default_values=1.0, number_of_elements=1)),
        ("SetResolution", smproperty.intvector(name="Resolution", default_values=256, number_of_elements=1)),
    ]
    for meth, d in deco:
        if names is not None and meth not in names:
            continue
        base = getattr(_MeshoidImageFilter, meth)

        def wrapper(self, *a, _b=base):
            return _b(self, *a)
        wrapper.__name__ = meth
        setattr(cls, meth, d(wrapper))
    return cls


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
            # per-particle channels don't depend on the view: build once per dataset/field/mode
            fa = pd.GetPointData().GetArray(name)
            key = ("chans", mode, name, fa.GetAddressAsString("vtkDataArray"), fa.GetMTime()) + _points_key(pd)
            if key not in _VIEW_CACHE:
                vol = None
                if mode == "LineIntegral":
                    rho = _arr(pd, "Density")
                    vol = m / rho if (m is not None and rho is not None) else (4 * np.pi / 3) * h**3 / DES_NGB
                _VIEW_CACHE.put(key, mo.projection_channels(mode, f, m, vol))
            chans, tail = _VIEW_CACHE.get(key)
            img = mo.project_channels(mode, chans, tail, pos, h, center, size, self._res, self._normal,
                                      self._up, self._smooth, self._depth)
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

    @smproperty.xml(_bool_xml(
        "AddLog10", "SetAddLog10", 1,
        "Also output log10_[array] (empty voxels set one decade below the smallest positive value). "
        "Use it for volume rendering and contours: ParaView's volume mapper ignores log-scaled colour maps."))
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

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        info = outInfoVec.GetInformationObject(0)
        r = self._res - 1
        info.Set(SDDP.WHOLE_EXTENT(), 0, r, 0, r, 0, r)
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
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
            img.GetPointData().AddArray(vtkconv.to_vtk_array(
                np.ravel(mo.safe_log10(vals), order="F"), f"log10_{label}"))
        out = vtkImageData.GetData(outInfoVec, 0)
        out.ShallowCopy(img)
        vtkconv.add_field_data(out, {"WindowCenter": center, "WindowSize": size})
        return 1


def _replace_block(out, name, fn):
    """Replace the point set named `name` in composite `out` by fn(shallow copy); True if found."""
    def fresh(leaf):
        new = leaf.NewInstance()
        new.ShallowCopy(leaf)  # never add arrays to the upstream object
        return fn(new)

    if isinstance(out, vtkPointSet):
        return None  # handled by the caller
    if isinstance(out, vtkPartitionedDataSetCollection):
        n = out.GetNumberOfPartitionedDataSets()
        for i in range(n):
            md = out.GetMetaData(i) if out.HasMetaData(i) else None
            if (md is not None and md.Get(vtkCompositeDataSet.NAME()) == name) or n == 1:
                pds = out.GetPartitionedDataSet(i)
                for k in range(pds.GetNumberOfPartitions()):
                    leaf = pds.GetPartition(k)
                    if isinstance(leaf, vtkPointSet) and leaf.GetNumberOfPoints():
                        pds.SetPartition(k, fresh(leaf))
                        return True
        return False
    if isinstance(out, vtkMultiBlockDataSet):
        for i in range(out.GetNumberOfBlocks()):
            md = out.GetMetaData(i) if out.HasMetaData(i) else None
            if md is not None and md.Get(vtkCompositeDataSet.NAME()) == name:
                leaf = out.GetBlock(i)
                if isinstance(leaf, vtkPointSet):
                    out.SetBlock(i, fresh(leaf))
                    return True
        return False
    return False


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
            add(out)
        elif not _replace_block(out, self._block, add):
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

