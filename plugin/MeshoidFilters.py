"""ParaView filters: kernel-weighted projections and slices of GIZMO particles via meshoid.

Input: the GIZMO reader's output (or any point set / composite of point sets);
the particle type is chosen by block name. Output: a 2D vtkImageData placed in
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
    vtkPartitionedDataSetCollection, vtkPointSet)
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP  # noqa: E402

from pvgizmo import meshoid_ops as mo  # noqa: E402  (imports pvgizmo first: multiprocessing fix)
from pvgizmo import vtkconv  # noqa: E402

DES_NGB = 32  # meshoid / GIZMO default neighbour number, for volumes when Density is missing
_CACHE = mo.MeshoidCache(maxsize=2)  # shared by all filter instances (KD-trees are big)

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


def _arr(pd, name):
    a = pd.GetPointData().GetArray(name)
    return None if a is None else ns.vtk_to_numpy(a)


class _MeshoidImageFilter(VTKPythonAlgorithmBase):
    """Shared plumbing: input block lookup, window, image output."""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData")
        self._block = "Gas"
        self._normal = [0.0, 0.0, 1.0]
        self._up = [0.0, 1.0, 0.0]
        self._auto = True
        self._center = [0.0, 0.0, 0.0]
        self._size = 1.0
        self._res = 256

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

    def _window(self, pos):
        if self._auto:
            return mo.auto_extent(pos)
        return np.array(self._center, dtype=np.float64), self._size

    def _kernel_radius(self, pd, pos, masses):
        h = _arr(pd, "SmoothingLength")
        if h is not None:
            return h
        key = ("H", id(pd.GetPoints().GetData()), pd.GetPoints().GetMTime(), len(pos))
        return _CACHE.get(key, pos, masses).SmoothingLength()  # stars/DM: meshoid kernel radius

    def _emit(self, outInfoVec, values, name, center, size):
        origin, spacing, direction = mo.image_geometry(center, size, self._res, self._normal, self._up)
        values = np.asarray(values)[:, :, None]  # explicit z axis: (res, res, k) is a 2D vector image
        img = vtkconv.image_from_array(values, origin, spacing, direction, name=name)
        out = vtkImageData.GetData(outInfoVec, 0)
        out.ShallowCopy(img)
        vtkconv.add_field_data(out, {"WindowCenter": center, "WindowSize": size,
                                     "Normal": self._normal, "ViewUp": self._up})


def _common_props(cls):
    """Attach the shared properties (decorators must be applied per proxy class)."""
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
        name = self._field_name() or "Masses"
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        f = _arr(pd, name)
        if f is None:
            raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
        m = _arr(pd, "Masses")
        h = self._kernel_radius(pd, pos, m)
        mode = mo.MODES[self._mode]
        vol = None
        if mode == "LineIntegral":
            rho = _arr(pd, "Density")
            vol = m / rho if (m is not None and rho is not None) else (4 * np.pi / 3) * h**3 / DES_NGB
        if mode == "MassWeightedAverage" and m is None:
            raise RuntimeError("Mass-weighted average needs a Masses array")
        center, size = self._window(pos)
        self.UpdateProgress(0.1)
        img = mo.project(mode, pos, h, center, size, self._res, normal=self._normal, view_up=self._up,
                         field=f, masses=m, volume=vol, smooth_fac=self._smooth, depth=self._depth,
                         conservative=self._conservative)
        self._emit(outInfoVec, img, f"{mode}({name})", center, size)
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

    @smproperty.xml(_bool_xml("SlopeLimiter", "SetSlopeLimiter", 0,
                              "Barth-Jespersen limiter on the gradient (as in GIZMO), order >= 1."))
    def SetSlopeLimiter(self, v):
        self._set("_limiter", bool(v))

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
        name = self._field_name() or "Density"
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        f = _arr(pd, name)
        if f is None:
            raise RuntimeError(f"array '{name}' not found in block '{self._block}'")
        m = _arr(pd, "Masses")
        key = ("slice", id(pd.GetPoints().GetData()), pd.GetPoints().GetMTime(), len(pos))
        M = _CACHE.get(key, pos, m, _arr(pd, "SmoothingLength"))
        center, size = self._window(pos)
        self.UpdateProgress(0.1)
        img = mo.slice_field(M, f, center, size, self._res, self._normal, self._up,
                             order=self._order, slope_limiter=self._limiter)
        self._emit(outInfoVec, img, f"Slice({name})", center, size)
        return 1

