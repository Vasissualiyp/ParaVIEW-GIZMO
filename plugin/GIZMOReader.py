"""ParaView reader for GIZMO HDF5 snapshots (single/multi-part, time series).

Output: vtkPartitionedDataSetCollection, one vtkPolyData partition per selected
particle type (block names: Gas, DarkMatter, Type2, Type3, Stars, Type5).
"""

import os
import sys

_ROOT = os.environ.get("PV_GIZMO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from paraview.util.vtkAlgorithm import (  # noqa: E402
    VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy)
from vtkmodules.vtkCommonCore import vtkDataArraySelection, vtkOutputWindow  # noqa: E402
from vtkmodules.vtkCommonDataModel import (  # noqa: E402
    vtkCompositeDataSet, vtkDataAssembly, vtkPartitionedDataSet, vtkPartitionedDataSetCollection)
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP  # noqa: E402

from pvgizmo import io, vtkconv  # noqa: E402  (pvgizmo import fixes multiprocessing first)

# Arrays enabled by default; everything else starts unchecked (big snapshots).
DEFAULT_ARRAYS = {"Masses", "Density", "SmoothingLength", "Velocities", "InternalEnergy",
                  "Temperature", "MagneticField",
                  # starlight (GIZMO Starlight Image / Star Properties): ages, metals (dust), sink L
                  "StellarFormationTime", "Metallicity", "StarLuminosity_Solar"}
RECENTER_NONE, RECENTER_PEAK, RECENTER_CUSTOM, RECENTER_STAR = 0, 1, 2, 3
HEADER_FIELDS = ("Time", "Redshift", "HubbleParam", "BoxSize", "Omega_Matter", "Omega_Lambda", "Omega0", "OmegaLambda",
                 "UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS")


def float32_collapses(pos):
    """True if the extent of `pos` is within ~1000 float32 steps at its distance from the origin."""
    if len(pos) < 2:
        return False
    extent = float(np.max(pos.max(0) - pos.min(0)))
    return extent < 1e3 * float(np.spacing(np.float32(np.abs(pos).max())))


def _warn_if_float32_collapses(pos, name):
    if float32_collapses(pos):
        vtkOutputWindow.GetInstance().DisplayWarningText(
            f"GIZMO reader: {name} spans less than ~1000 float32 steps at its distance from the "
            "origin and will render collapsed. Set Recenter to 'Gas density peak' or 'Custom center'.")


def _selection_observer(alg):
    return lambda *args: alg.Modified()


@smproxy.reader(name="GIZMOSnapshotReader", label="GIZMO Snapshot Reader",
                extensions="hdf5 h5", file_description="GIZMO snapshot files")
class GIZMOSnapshotReader(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1,
                                        outputType="vtkPartitionedDataSetCollection")
        self._filenames = []
        self._snaps = None  # list of (time, parts, header), sorted by time
        self._types = vtkDataArraySelection()
        self._arrays = vtkDataArraySelection()
        for s in (self._types, self._arrays):
            s.AddObserver("ModifiedEvent", _selection_observer(self))
        self._stride = 1
        self._physical = True
        self._use_bbox = False
        self._recenter = RECENTER_NONE
        self._center = [0.0, 0.0, 0.0]
        self._center_id = 0
        self._centered_on = -1
        self._peak_cache = {}  # (first part, physical) -> density-peak position
        self._bbox = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    # ---------------------------------------------------------------- properties
    @smproperty.stringvector(name="FileNames", label="File Names", animateable="1",
                             clean_command="RemoveAllFileNames", command="AddFileName",
                             repeat_command="1", number_of_elements_per_command="1",
                             panel_visibility="never")
    @smdomain.filelist()
    @smhint.filechooser(extensions="hdf5 h5", file_description="GIZMO snapshot files")
    def AddFileName(self, name):
        if name and name not in self._filenames:
            self._filenames.append(name)
            self._snaps = None
            self.Modified()

    def RemoveAllFileNames(self):
        if self._filenames:
            self._filenames = []
            self._snaps = None
            self.Modified()

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        snaps = self._scan()
        return [s[0] for s in snaps] if len(snaps) > 1 else None

    @smproperty.dataarrayselection(name="ParticleTypes")
    def GetParticleTypeSelection(self):
        return self._types

    @smproperty.dataarrayselection(name="PointArrays")
    def GetPointArraySelection(self):
        return self._arrays

    @smproperty.intvector(name="Stride", default_values=1)
    @smdomain.intrange(min=1, max=1000)
    def SetStride(self, n):
        if int(n) != self._stride:
            self._stride = max(1, int(n))
            self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="PhysicalUnits" command="SetPhysicalUnits" number_of_elements="1"
                           default_values="1">
          <BooleanDomain name="bool"/>
          <Documentation>Apply the scale-factor and h conversions of load_from_snapshot
          (cosmological runs). Off: raw code units.</Documentation>
        </IntVectorProperty>""")
    def SetPhysicalUnits(self, v):
        if bool(v) != self._physical:
            self._physical = bool(v)
            self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="Recenter" command="SetRecenter" number_of_elements="1"
                           default_values="0">
          <EnumerationDomain name="enum">
            <Entry value="0" text="None"/>
            <Entry value="1" text="Gas density peak"/>
            <Entry value="2" text="Custom center"/>
            <Entry value="3" text="Most massive star"/>
          </EnumerationDomain>
          <Documentation>Subtract a center from all coordinates (in double precision) before
          they reach VTK. Opt-in. Rendering is float32: a zoom region far from the box origin
          can be smaller than float32 spacing there and collapses to a point without this (the
          reader warns when that happens). The
          offset is stored in the CenterOffset field array. With a time series the density
          peak / star is found per snapshot, so a movie stays centred on it. "Most massive
          star": the most massive sink (PartType5), else star particle (PartType4), else the
          density peak; set CenterParticleID to follow one particular star instead.</Documentation>
        </IntVectorProperty>""")
    def SetRecenter(self, v):
        if int(v) != self._recenter:
            self._recenter = int(v)
            self.Modified()

    @smproperty.doublevector(name="Center", default_values=[0, 0, 0])
    def SetCenter(self, x, y, z):
        if [x, y, z] != self._center:
            self._center = [x, y, z]
            self.Modified()

    @smproperty.xml("""
        <IdTypeVectorProperty name="CenterParticleID" command="SetCenterParticleID"
                              number_of_elements="1" default_values="0">
          <Documentation>Recenter = Most massive star: 0 centres on the most massive star of
          each snapshot; a ParticleIDs value follows that star (no jumps when another one
          becomes the most massive). The ID used is in the CenterParticleID field array.
          </Documentation>
        </IdTypeVectorProperty>""")
    def SetCenterParticleID(self, v):
        if int(v) != self._center_id:
            self._center_id = int(v)
            self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="UseBoundingBox" command="SetUseBoundingBox" number_of_elements="1"
                           default_values="0">
          <BooleanDomain name="bool"/>
        </IntVectorProperty>""")
    def SetUseBoundingBox(self, v):
        if bool(v) != self._use_bbox:
            self._use_bbox = bool(v)
            self.Modified()

    @smproperty.doublevector(name="BoundingBox", default_values=[-1, 1, -1, 1, -1, 1])
    def SetBoundingBox(self, x0, x1, y0, y1, z0, z1):
        b = [x0, x1, y0, y1, z0, z1]
        if b != self._bbox:
            self._bbox = b
            self.Modified()

    # ---------------------------------------------------------------- pipeline
    def _scan(self):
        """Group FileNames into snapshots, read headers and field lists (cached)."""
        if self._snaps is not None:
            return self._snaps
        seen, snaps = set(), []
        for f in self._filenames:
            key = io.snapshot_key(f)
            if key in seen:
                continue
            seen.add(key)
            parts = io.snapshot_parts(f)
            snaps.append((float(io.read_header(parts)["Time"]), parts))
        snaps.sort(key=lambda s: s[0])
        self._snaps = snaps
        if snaps:  # populate selections from the first snapshot, keeping user choices
            for t, fields in io.list_fields(snaps[0][1]).items():
                name = io.PTYPE_NAMES[t]
                if not self._types.ArrayExists(name):
                    self._types.AddArray(name, True)
                for k in fields:
                    if k != "Coordinates" and not self._arrays.ArrayExists(k):
                        self._arrays.AddArray(k, k in DEFAULT_ARRAYS)
        return snaps

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        snaps = self._scan()
        info = outInfoVec.GetInformationObject(0)
        info.Remove(SDDP.TIME_STEPS())
        info.Remove(SDDP.TIME_RANGE())
        if len(snaps) > 1:
            times = [s[0] for s in snaps]
            info.Set(SDDP.TIME_STEPS(), times, len(times))
            info.Set(SDDP.TIME_RANGE(), [times[0], times[-1]], 2)
        return 1

    def _pick_snapshot(self, outInfo):
        snaps = self._scan()
        if len(snaps) > 1 and outInfo.Has(SDDP.UPDATE_TIME_STEP()):
            t = outInfo.Get(SDDP.UPDATE_TIME_STEP())
            return snaps[int(np.argmin([abs(s[0] - t) for s in snaps]))]
        return snaps[0]

    def _offset(self, parts):
        if self._recenter == RECENTER_CUSTOM:
            return np.array(self._center, dtype=np.float64)
        if self._recenter == RECENTER_STAR:
            key = ("star", parts[0], self._physical, self._center_id)
            if key not in self._peak_cache:
                self._peak_cache[key] = io.star_center(parts, physical=self._physical,
                                                       particle_id=self._center_id)
            hit = self._peak_cache[key]
            if hit is not None:
                self._centered_on = hit[1]
                return hit[0]
            self._centered_on = -1  # no stars yet: fall back to the gas density peak
        if self._recenter in (RECENTER_PEAK, RECENTER_STAR):
            key = (parts[0], self._physical)
            if key not in self._peak_cache:
                self._peak_cache[key] = io.density_peak(parts, physical=self._physical)
            return self._peak_cache[key]
        return None

    def RequestData(self, request, inInfoVec, outInfoVec):
        if not self._filenames:
            return 1
        outInfo = outInfoVec.GetInformationObject(0)
        time, parts = self._pick_snapshot(outInfo)
        hdr = io.read_header(parts)
        fields = io.list_fields(parts)
        # Parallel pvserver (mpiexec -n P): every rank executes the reader. Read on piece 0
        # only, so data isn't duplicated P times; the other ranks keep the block structure
        # with empty partitions. (One process already uses all cores via numba threads.)
        piece = outInfo.Get(SDDP.UPDATE_PIECE_NUMBER()) if outInfo.Has(SDDP.UPDATE_PIECE_NUMBER()) else 0
        offset = self._offset(parts) if piece == 0 else None
        out = vtkPartitionedDataSetCollection.GetData(outInfoVec, 0)
        out.Initialize()  # the output object is reused across executions
        assembly = vtkDataAssembly()
        # Root must be "Root": ExtractBlock & co. default to the "Hierarchy" (/Root/<name>)
        # or to this "Assembly" depending on whether the reader already ran, so the
        # same selector (/Root/Gas) has to resolve in both.
        assembly.SetRootNodeName("Root")

        types = [t for t in fields if self._types.ArrayIsEnabled(io.PTYPE_NAMES[t])]
        out.SetNumberOfPartitionedDataSets(len(types))
        for idx, t in enumerate(types):
            self.UpdateProgress(idx / max(len(types), 1))
            names = [k for k in fields[t] if k != "Coordinates" and self._arrays.ArrayIsEnabled(k)]
            data = {} if piece != 0 else io.read_ptype(
                parts, t, names, physical=self._physical, stride=self._stride,
                bbox=self._bbox if self._use_bbox else None, center=offset)
            pos = data.pop("Coordinates", np.zeros((0, 3), np.float32))
            if offset is None and piece == 0:
                _warn_if_float32_collapses(pos, io.PTYPE_NAMES[t])
            if "Density" in data:  # opacity/colour curves are linear: give them log density
                d = np.asarray(data["Density"], dtype=np.float64)
                tiny = d[d > 0].min() if np.any(d > 0) else 1.0
                data["log10_Density"] = np.log10(np.maximum(d, tiny)).astype(np.float32)
            if "SmoothingLength" in data:  # VTK SPH kernels: support = 2h, GIZMO H = support
                data["SmoothingLengthVTK"] = 0.5 * data["SmoothingLength"]
            pd = vtkconv.particles_to_polydata(pos, data)
            pds = vtkPartitionedDataSet()
            pds.SetPartition(0, pd)
            out.SetPartitionedDataSet(idx, pds)
            out.GetMetaData(idx).Set(vtkCompositeDataSet.NAME(), io.PTYPE_NAMES[t])
            node = assembly.AddNode(io.PTYPE_NAMES[t])
            assembly.AddDataSetIndex(node, idx)
        out.SetDataAssembly(assembly)

        meta = {k: hdr[k] for k in HEADER_FIELDS if k in hdr}
        meta["Cosmological"] = int(io.is_cosmological(hdr))
        meta["PhysicalUnits"] = int(self._physical)
        meta["SnapshotNumber"] = io.snapshot_number(parts[0]) if io.snapshot_number(parts[0]) is not None else -1
        meta["NumPart_Total"] = hdr["NumPart_Total"]
        meta["CenterOffset"] = offset if offset is not None else np.zeros(3)
        if self._recenter == RECENTER_STAR:
            meta["CenterParticleID"] = self._centered_on
        vtkconv.add_field_data(out, meta)  # DATA_TIME_STEP is set by the executive
        return 1
