"""ParaView filters for a "space look": starlight images, two-field composites, star properties.

* Starlight Image: FIRE-style mock three-band image of the stars, dimmed and
  reddened by gas metals along each sightline (pfh_python's ray-traced stellar
  images). Follows the camera like Meshoid Projection.
* Meshoid Composite: vizmo's two-field image; brightness from one projection
  (column density), colour from another (e.g. mass-weighted temperature).
* Star Properties: per-star age, band luminosities, a display colour and size,
  for glowing-star Point Gaussian rendering (pvgizmo.pvhelpers.show_stars).

The image filters output an RGB (uchar) array: show it with Map Scalars off.
"""

import os
import sys

_ROOT = os.environ.get("PV_GIZMO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy  # noqa: E402
from vtkmodules.util import numpy_support as ns  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkPointSet  # noqa: E402

from pvgizmo import live as _live  # noqa: E402  (imports pvgizmo first: multiprocessing fix)
from pvgizmo import looks, stars  # noqa: E402
from pvgizmo import meshoid_ops as mo  # noqa: E402
from pvgizmo.pvbase import (  # noqa: E402
    _VIEW_CACHE, MODE_LABELS, _arr, _bool_xml, _common_props, _meshoid_for, _MeshoidImageFilter,
    _parallel_piece, _points_key, _replace_block, _resolve_field, find_block, project_field)

BAND_LABELS = tuple(" ".join(b) for b in stars.BAND_SETS)


def _enum_xml(name, command, labels, default=0, doc=""):
    entries = "".join(f'<Entry value="{i}" text="{t}"/>' for i, t in enumerate(labels))
    return f"""
<IntVectorProperty name="{name}" command="{command}" number_of_elements="1" default_values="{default}">
  <EnumerationDomain name="enum">{entries}</EnumerationDomain><Documentation>{doc}</Documentation>
</IntVectorProperty>"""


def _array_xml(name, idx, default, doc):
    return f"""
<StringVectorProperty name="{name}" label="{name}" command="SetInputArrayToProcess"
                      number_of_elements="5" element_types="0 0 0 0 2"
                      default_values_delimiter=";" default_values="{idx};0;0;0;{default}">
  <ArrayListDomain name="array_list" input_domain_name="inputs_array">
    <RequiredProperties><Property name="Input" function="Input"/></RequiredProperties>
  </ArrayListDomain>
  <Documentation>{doc}</Documentation>
</StringVectorProperty>"""


def _meta(obj):
    """Snapshot header values the reader stored as field data ({name: float or array})."""
    fd = obj.GetFieldData() if obj is not None else None
    out = {}
    if fd is None:
        return out
    for i in range(fd.GetNumberOfArrays()):
        a = fd.GetArray(i)
        if a is not None and a.GetName():
            out[a.GetName()] = ns.vtk_to_numpy(a)
    return out


def _star_light(pd, bands, meta):
    """(luminosities (N, 3), ages or None) for one star block, cached per dataset and band set."""
    key = ("starlight", tuple(bands)) + _points_key(pd)
    if key not in _VIEW_CACHE:
        m = _arr(pd, "Masses")
        if m is None:
            raise RuntimeError("star particles need a Masses array")
        sft = _arr(pd, "StellarFormationTime")
        lum_solar = _arr(pd, "StarLuminosity_Solar")
        if lum_solar is None:
            lum_solar = _arr(pd, "StarLuminosity")
        if lum_solar is not None:
            sft = None  # individual stars (STARFORGE sinks): their own luminosity
        _VIEW_CACHE.put(key, stars.star_luminosities(bands, m, meta, sft=sft, z=stars.metallicity(
            _arr(pd, "Metallicity")), lum_solar=lum_solar))
    return _VIEW_CACHE.get(key)


def _blocks(spec):
    return [b.strip() for b in str(spec).split(",") if b.strip()]


@smproxy.filter(name="GIZMOStarlightImage", label="GIZMO Starlight Image")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props(["SetNormal", "SetViewUp", "SetAutoWindow", "SetCenter", "SetSize", "SetResolution"])
class GIZMOStarlightImage(_MeshoidImageFilter):
    """Mock three-band image of the starlight, attenuated by dust along each line of sight.

    Star particles shine with SSP luminosities (age, metallicity; BC03 Chabrier)
    or, for individual stars, their StarLuminosity_Solar; gas metals absorb with
    per-band opacities. Output: RGB (uchar, show with Map Scalars off) and the
    attenuated surface brightness per band, Starlight_[band] (L_sun / length^2).
    """

    def __init__(self):
        _MeshoidImageFilter.__init__(self)
        self._stars, self._gas = "Stars", "Gas"
        self._bands = 0
        self._extinction = True
        self._dust = 1.0
        self._dynrange = 1e4
        self._saturation = 1e-4
        self._scheme = 0
        self._hscale = 3.0
        self._hmax_frac = 0.01
        self._depth = 0.0

    @smproperty.stringvector(name="StarBlocks", default_values="Stars", number_of_elements=1)
    @smdomain.xml("<Documentation>Comma-separated blocks to treat as stars, e.g. 'Stars' (FIRE) or "
                  "'Type5' (STARFORGE sinks).</Documentation>")
    def SetStarBlocks(self, s):
        self._set("_stars", str(s))

    @smproperty.stringvector(name="GasBlock", default_values="Gas", number_of_elements=1)
    def SetGasBlock(self, s):
        self._set("_gas", str(s))

    @smproperty.xml(_enum_xml("Bands", "SetBands", BAND_LABELS, 0,
                              "Blue, green, red bands. u g r is the FIRE default."))
    def SetBands(self, b):
        self._set("_bands", int(b))

    @smproperty.xml(_bool_xml("Extinction", "SetExtinction", 1, "Dust attenuation by the gas in front of each star."))
    def SetExtinction(self, v):
        self._set("_extinction", bool(v))

    @smproperty.doublevector(name="DustToGasScale", default_values=1.0)
    def SetDustToGasScale(self, s):
        self._set("_dust", float(s))

    @smproperty.doublevector(name="DynamicRange", default_values=1e4)
    def SetDynamicRange(self, d):
        self._set("_dynrange", max(float(d), 1.0001))

    @smproperty.doublevector(name="Saturation", default_values=1e-4)
    @smdomain.xml("<Documentation>Fraction of the pixels that come out saturated (sets the brightness).</Documentation>")
    def SetSaturation(self, s):
        self._set("_saturation", float(s))

    @smproperty.xml(_enum_xml("ColorScheme", "SetColorScheme", ("NASA (log per band)", "SDSS (asinh)"), 0))
    def SetColorScheme(self, s):
        self._set("_scheme", int(s))

    @smproperty.doublevector(name="StarKernelScale", default_values=3.0)
    @smdomain.xml("<Documentation>Star smoothing = this x the star-star kernel radius (pfh_python: 3), "
                  "softly capped at StarKernelMax x the window size.</Documentation>")
    def SetStarKernelScale(self, s):
        self._set("_hscale", float(s))

    @smproperty.doublevector(name="StarKernelMax", default_values=0.01)
    def SetStarKernelMax(self, s):
        self._set("_hmax_frac", float(s))

    @smproperty.doublevector(name="Depth", default_values=0.0)
    def SetDepth(self, d):
        self._set("_depth", float(d))

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 1,
        "GUI: keep the image aimed at the render view's camera, updating as you rotate/zoom/pan."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)

    def _sources(self, inp, bands, meta):
        """Star positions, raw kernel radii, luminosities (view-independent, cached)."""
        pds = [pd for pd in (find_block(inp, n) for n in _blocks(self._stars))
               if pd is not None and pd.GetNumberOfPoints()]
        if not pds:
            return None
        key = ("sources", tuple(bands)) + tuple(_points_key(pd) for pd in pds)
        if key not in _VIEW_CACHE:
            _VIEW_CACHE.put(key, self._gather_sources(pds, bands, meta))
        return _VIEW_CACHE.get(key)

    def _gather_sources(self, pds, bands, meta):
        pos, h, lum = [], [], []
        for pd in pds:
            p = ns.vtk_to_numpy(pd.GetPoints().GetData())
            L, _ = _star_light(pd, bands, meta)
            key = ("starh",) + _points_key(pd)
            if key not in _VIEW_CACHE:
                hh = _arr(pd, "SmoothingLength")
                if hh is None:
                    # too few for a neighbour search: infinite -> the soft cap sets the size
                    hh = _meshoid_for(pd, p).SmoothingLength() if len(p) > 64 else np.full(len(p), np.inf)
                _VIEW_CACHE.put(key, np.asarray(hh, np.float64))
            pos.append(p)
            h.append(_VIEW_CACHE.get(key))
            lum.append(L)
        return np.concatenate(pos).astype(np.float64), np.concatenate(h), np.concatenate(lum)

    def _absorbers(self, inp, meta):
        pd = find_block(inp, self._gas)
        if pd is None or pd.GetNumberOfPoints() == 0:
            return None
        key = ("absorbers",) + _points_key(pd)
        if key not in _VIEW_CACHE:
            pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
            m = _arr(pd, "Masses")
            if m is None:
                return None
            h = self._kernel_radius(pd, pos, m)
            z = stars.metallicity(_arr(pd, "Metallicity"))
            zfac = 1.0 if z is None else np.asarray(z, np.float64) / stars.ZSUN
            _VIEW_CACHE.put(key, (pos.astype(np.float64), np.asarray(h, np.float64), m * zfac))
        return _VIEW_CACHE.get(key)

    def RequestData(self, request, inInfoVec, outInfoVec):
        inp = vtkDataObject.GetData(inInfoVec[0], 0)
        meta = _meta(inp)
        bands = stars.BAND_SETS[self._bands]
        src = self._sources(inp, bands, meta)
        if src is None:
            if _parallel_piece(inInfoVec):
                return 1
            raise RuntimeError(f"no star particles in blocks '{self._stars}'")
        spos, sh, slum = src
        if self._auto:
            center, size = mo.auto_extent(spos)
        else:
            center, size = np.array(self._center, dtype=np.float64), self._size
        hmax = self._hmax_frac * size
        with np.errstate(divide="ignore"):
            sh_eff = 1.0 / (1.0 / (self._hscale * sh) + 1.0 / hmax)  # pfh_python's soft cap
        gas = self._absorbers(inp, meta) if self._extinction else None
        # stars + gas in one set of arrays, cached: per frame only the star kernels change
        key = ("rayinputs", id(spos), id(gas), self._dust, len(spos))
        if key not in _VIEW_CACHE:
            if gas is None:
                x, h, mext, lum = spos, sh_eff.copy(), np.zeros(len(spos)), slum
            else:
                gx, gh, gm = gas
                x = np.concatenate([spos, gx])
                h = np.concatenate([sh_eff, gh])
                mext = np.concatenate([np.zeros(len(spos)), gm * self._dust])
                lum = np.concatenate([slum, np.zeros((len(gx), 3))])
            _VIEW_CACHE.put(key, (x, h, mext, lum, (mext > 0) | np.any(lum != 0, axis=1), (spos, gas)))
        x, h, mext, lum, active, _ = _VIEW_CACHE.get(key)  # (spos, gas) kept alive: ids stay unique
        h[:len(spos)] = sh_eff
        kappa = np.array([stars.kappa_code(b, meta) * k for b, k in zip(bands, stars.KAPPA_TWEAK)])
        self.UpdateProgress(0.1)
        basis = mo.view_basis(self._normal, self._up)
        maps, _ = stars.raytrace(x, h, mext, lum, center, basis, size, self._res, kappa, depth=self._depth,
                                 active=active)
        maxden, minden = stars.threeband_levels(maps, self._saturation, self._dynrange)
        rgb = stars.threeband_rgb(maps, maxden, minden, scheme=("nasa", "sdss")[self._scheme])
        extra = {f"Starlight_{b}": maps[:, :, k] for k, b in enumerate(bands)}
        self._emit(outInfoVec, rgb, "RGB", center, size, extra=extra,
                   meta={"BrightnessLevels": [minden, maxden]})
        _live.debug_log("exec", filter="starlight", normal=list(self._normal), res=self._res)
        return 1


@smproxy.filter(name="MeshoidComposite", label="Meshoid Composite")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
@_common_props
class MeshoidComposite(_MeshoidImageFilter):
    """Two projections in one image (vizmo's composite): brightness from Field, colour from Field2.

    Default: column density (log) sets the lightness, mass-weighted temperature (log)
    picks the colour from Colormap. Output: RGB (uchar, show with Map Scalars off) and
    both maps. Ranges: [0, 0] = automatic (percentiles of each map).
    """

    def __init__(self):
        _MeshoidImageFilter.__init__(self)
        self._mode, self._mode2 = 0, 2
        self._cmap = 0
        self._llog, self._hlog = True, True
        self._lrange, self._hrange = [0.0, 0.0], [0.0, 0.0]
        self._depth = 0.0

    @smproperty.xml(_array_xml("Field", 0, "Masses", "Lightness: projected with Mode."))
    def SetInputArrayToProcess(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    # one property per method: the second array property lives here, but its XML command is
    # SetInputArrayToProcess (index 1), so this method itself is never called
    @smproperty.xml(_array_xml("Field2", 1, "Temperature", "Colour: projected with Mode2."))
    def _Field2(self, *args):
        return VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    @smproperty.xml(_enum_xml("Mode", "SetMode", MODE_LABELS, 0))
    def SetMode(self, m):
        self._set("_mode", int(m))

    @smproperty.xml(_enum_xml("Mode2", "SetMode2", MODE_LABELS, 2))
    def SetMode2(self, m):
        self._set("_mode2", int(m))

    @smproperty.xml(_enum_xml("Colormap", "SetColormap", looks.COLORMAPS, 0))
    def SetColormap(self, c):
        self._set("_cmap", int(c))

    @smproperty.xml(_bool_xml("LogLightness", "SetLogLightness", 1))
    def SetLogLightness(self, v):
        self._set("_llog", bool(v))

    @smproperty.xml(_bool_xml("LogColor", "SetLogColor", 1))
    def SetLogColor(self, v):
        self._set("_hlog", bool(v))

    @smproperty.doublevector(name="LightnessRange", default_values=[0, 0], number_of_elements=2)
    def SetLightnessRange(self, lo, hi):
        self._set("_lrange", [float(lo), float(hi)])

    @smproperty.doublevector(name="ColorRange", default_values=[0, 0], number_of_elements=2)
    def SetColorRange(self, lo, hi):
        self._set("_hrange", [float(lo), float(hi)])

    @smproperty.doublevector(name="Depth", default_values=0.0)
    def SetDepth(self, d):
        self._set("_depth", float(d))

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 1,
        "GUI: keep the image aimed at the render view's camera, updating as you rotate/zoom/pan."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)

    def RequestData(self, request, inInfoVec, outInfoVec):
        pd = self._particles(inInfoVec)
        if pd is None:
            return 1
        names = [self.GetInputArrayInformation(k).Get(vtkDataObject.FIELD_NAME()) for k in (0, 1)]
        n1 = _resolve_field(pd, names[0], "Masses")
        fallback = next((n for n in ("Temperature", "InternalEnergy", "Density") if _arr(pd, n) is not None), n1)
        n2 = _resolve_field(pd, names[1], fallback)
        pos = ns.vtk_to_numpy(pd.GetPoints().GetData())
        h = self._kernel_radius(pd, pos, _arr(pd, "Masses"))
        center, size = self._window(pos, pd)
        self.UpdateProgress(0.1)
        m1, m2 = mo.MODES[self._mode], mo.MODES[self._mode2]
        args = (pos, h, center, size, self._res, self._normal, self._up, 1.0, self._depth)
        a = project_field(pd, n1, m1, *args)
        b = project_field(pd, n2, m2, *args)
        a = np.linalg.norm(a, axis=-1) if a.ndim == 3 else a  # vectors: magnitude
        b = np.linalg.norm(b, axis=-1) if b.ndim == 3 else b
        rng = lambda r: tuple(r) if r[1] > r[0] else None  # noqa: E731
        rgb, lr, hr = looks.hsv_composite(a, b, looks.COLORMAPS[self._cmap], rng(self._lrange), rng(self._hrange),
                                          self._llog, self._hlog)
        self._emit(outInfoVec, rgb, "RGB", center, size, extra={f"{m1}_{n1}": a, f"{m2}_{n2}": b},
                   meta={"LightnessRange": lr, "ColorRange": hr})
        _live.debug_log("exec", filter="composite", normal=list(self._normal), res=self._res)
        return 1


@smproxy.filter(name="GIZMOStarProperties", label="GIZMO Star Properties")
@smhint.xml('<ShowInMenu category="GIZMO"/>')
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=True)
class GIZMOStarProperties(VTKPythonAlgorithmBase):
    """Adds per-star arrays to the star block; everything else passes through.

    StellarAge (Gyr; populations only), Luminosity_[band] (L_sun in band),
    Luminosity (sum of the three bands), StarRGB (uchar display colour from the band
    ratios) and StarSize (grows with log L, for Point Gaussian scaling).
    """

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkDataObject")
        self._block = "Stars"
        self._bands = 0
        self._size = 1.0
        self._ref = 0.0

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

    @smproperty.stringvector(name="Block", default_values="Stars", number_of_elements=1)
    def SetBlock(self, s):
        self._set("_block", str(s))

    @smproperty.xml(_enum_xml("Bands", "SetBands", BAND_LABELS, 0))
    def SetBands(self, b):
        self._set("_bands", int(b))

    @smproperty.doublevector(name="SizeScale", default_values=1.0)
    @smdomain.xml("<Documentation>StarSize = SizeScale x 0.01 x ReferenceSize x "
                  "(1 + 0.5 log10(L / median L)), clipped to 0.3-4x: the median star is ~1% of ReferenceSize, "
                  "and luminosities spanning many decades stay visible without swamping the view.</Documentation>")
    def SetSizeScale(self, s):
        self._set("_size", float(s))

    @smproperty.doublevector(name="ReferenceSize", default_values=0.0)
    @smdomain.xml("<Documentation>Length the star sizes are relative to, e.g. the width of the image "
                  "they overlay. 0: the extent of the stars themselves.</Documentation>")
    def SetReferenceSize(self, s):
        self._set("_ref", float(s))

    @smproperty.xml(_bool_xml(
        "FollowCamera", "SetFollowCamera", 1,
        "GUI: ReferenceSize = the width the view shows, so stars keep their on-screen size when you zoom."))
    def SetFollowCamera(self, v):
        self._follow = bool(v)

    def RequestData(self, request, inInfoVec, outInfoVec):
        inp = vtkDataObject.GetData(inInfoVec[0], 0)
        out = vtkDataObject.GetData(outInfoVec, 0)
        out.ShallowCopy(inp)
        meta = _meta(inp)
        bands = stars.BAND_SETS[self._bands]

        def add(pd):
            L, age = _star_light(pd, bands, meta)
            pdat = pd.GetPointData()
            from pvgizmo import vtkconv
            if age is not None:
                pdat.AddArray(vtkconv.to_vtk_array(age, "StellarAge"))
            for k, b in enumerate(bands):
                pdat.AddArray(vtkconv.to_vtk_array(L[:, k], f"Luminosity_{b}"))
            ltot = L.sum(axis=1)
            pdat.AddArray(vtkconv.to_vtk_array(ltot, "Luminosity"))
            pdat.AddArray(vtkconv.to_vtk_array(stars.star_rgb(L), "StarRGB"))
            if self._ref > 0:
                extent = self._ref
            else:
                _, extent = mo.auto_extent(ns.vtk_to_numpy(pd.GetPoints().GetData()))
            med = np.median(ltot[ltot > 0]) if np.any(ltot > 0) else 1.0
            with np.errstate(divide="ignore"):
                rel = np.clip(1.0 + 0.5 * np.log10(np.maximum(ltot, 1e-300) / med), 0.3, 4.0)
            size = self._size * 0.01 * extent * rel
            pdat.AddArray(vtkconv.to_vtk_array(size, "StarSize"))
            return pd

        if isinstance(out, vtkPointSet):
            if out.GetNumberOfPoints():
                add(out)
        elif not _replace_block(out, self._block, add) and not _parallel_piece(inInfoVec):
            raise RuntimeError(f"no particles in block '{self._block}'")
        return 1


# Client-server GUI: install the FollowCamera auto-linker on the client (see MeshoidFilters),
# and add the matplotlib colour maps ParaView lacks (client only; no-op elsewhere).
try:
    _live.enable_auto_follow()
    if _live.in_gui():
        looks.register_presets()
except Exception:
    pass
