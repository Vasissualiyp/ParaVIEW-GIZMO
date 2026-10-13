"""Shared plumbing of the GIZMO ParaView filters (plugin/*.py): block lookup, caches,
the 2D image-filter base class and its common properties.

Imported only inside ParaView (needs paraview.util.vtkAlgorithm). The caches are
module-level, so all plugin files share one Meshoid KD-tree per dataset.
"""

import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smproperty
from vtkmodules.util import numpy_support as ns
from vtkmodules.vtkCommonDataModel import (
    vtkCompositeDataSet, vtkDataObject, vtkMultiBlockDataSet, vtkPartitionedDataSet,
    vtkPartitionedDataSetCollection, vtkPointSet, vtkStructuredGrid)
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP

from pvgizmo import live as _live
from pvgizmo import meshoid_ops as mo
from pvgizmo import vtkconv

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


_VIEW_CACHE = _LRU(maxsize=12)  # camera moves reuse these: only the deposit runs per frame
# (starlight alone keeps ~5 entries per dataset: luminosities, star kernels, sources, absorbers, inputs)

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


def _parallel_piece(inInfoVec):
    """True when this execution is one of several pieces (parallel pvserver/pvbatch)."""
    info = inInfoVec[0].GetInformationObject(0)
    key = SDDP.UPDATE_NUMBER_OF_PIECES()
    return info.Has(key) and info.Get(key) > 1


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
        """The particle block, or None on an empty piece of a parallel (MPI) run."""
        pd = find_block(vtkDataObject.GetData(inInfoVec[0], 0), self._block)
        if pd is None or pd.GetNumberOfPoints() == 0:
            if _parallel_piece(inInfoVec):
                return None  # data lives on another rank (the reader loads on piece 0)
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

    def _emit(self, outInfoVec, values, name, center, size, extra=None, meta=None):
        """Write the map (plus `extra` {name: map} arrays and `meta` field data) as the output."""
        pts = mo.slice_points(center, size, self._res, self._normal, self._up).reshape(self._res, self._res, 3)
        grid = vtkconv.structured_grid_from_array(values, pts, name=name)
        for k, v in (extra or {}).items():
            vtkconv.add_map_array(grid, v, k)
        out = vtkStructuredGrid.GetData(outInfoVec, 0)
        out.ShallowCopy(grid)
        vtkconv.add_field_data(out, {"WindowCenter": center, "WindowSize": size,
                                     "Normal": self._normal, "ViewUp": self._up, **(meta or {})})


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


def project_field(pd, name, mode, pos, h, center, size, res, normal, up, smooth=1.0, depth=0.0):
    """Projection of point array `name` (mode one of mo.MODES) through the shared channel cache.

    The per-particle channels don't depend on the view, so camera moves only redo the deposit.
    """
    m = _arr(pd, "Masses")
    f = _arr(pd, name)
    if f is None:
        raise RuntimeError(f"array '{name}' not found")
    if mode == "MassWeightedAverage" and m is None:
        raise RuntimeError("Mass-weighted average needs a Masses array")
    fa = pd.GetPointData().GetArray(name)
    key = ("chans", mode, name, fa.GetAddressAsString("vtkDataArray"), fa.GetMTime()) + _points_key(pd)
    if key not in _VIEW_CACHE:
        vol = None
        if mode == "LineIntegral":
            rho = _arr(pd, "Density")
            vol = m / rho if (m is not None and rho is not None) else (4 * np.pi / 3) * h**3 / DES_NGB
        _VIEW_CACHE.put(key, mo.projection_channels(mode, f, m, vol))
    chans, tail = _VIEW_CACHE.get(key)
    return mo.project_channels(mode, chans, tail, pos, h, center, size, res, normal, up, smooth, depth)
