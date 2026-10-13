"""Space look: colour maps, two-field composites, and ParaView colour-map presets.

The composite is vizmo's (shaders/composite.wgsl): field 1 sets the lightness
(HSV value) and field 2 picks the hue from a colour map, so e.g. column
density is brightness and temperature is colour. A lookup table can't map two
scalars to one colour, so the result is an RGB array shown with Map Scalars off.
"""

import json
import os

import numpy as np

# vizmo's colour maps (vizmo/colormaps.py), by matplotlib name
COLORMAPS = ("magma", "inferno", "viridis", "plasma", "coolwarm", "hot", "bone", "cividis")
# the ParaView preset that matches each, or None if ParaView has no equivalent (generated)
PV_PRESETS = {"magma": "Magma (matplotlib)", "inferno": "Inferno (matplotlib)",
              "viridis": "Viridis (matplotlib)", "plasma": "Plasma (matplotlib)", "cividis": "Cividis"}
GENERATED = ("hot", "bone", "coolwarm", "afmhot", "gist_heat", "twilight_shifted", "cubehelix")


def colormap_table(name, n=256):
    """(n, 3) float RGB samples of a matplotlib colour map."""
    import matplotlib
    return np.asarray(matplotlib.colormaps[name](np.linspace(0.0, 1.0, n)))[:, :3]


def preset_name(name):
    """The ParaView preset name used for matplotlib map `name`."""
    return PV_PRESETS.get(name) or f"{name} (matplotlib, GIZMO)"


def normalize(values, lo, hi, log):
    """Map values to [0, 1] between lo and hi (log10 of the values if log); NaN stays NaN."""
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if log:
            v = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), -np.inf)
            lo, hi = np.log10(lo), np.log10(hi)
        t = (v - lo) / (hi - lo) if hi != lo else np.zeros_like(v)
    return np.where(np.isnan(v), np.nan, np.clip(t, 0.0, 1.0))


def auto_range(values, log, percentiles):
    """(lo, hi) from percentiles of the finite (positive, if log) values; None if there are none."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, percentiles)
    if hi <= lo:
        lo, hi = v.min(), v.max()
    if hi <= lo:
        hi = lo * 10 if (log and lo > 0) else lo + 1
    return float(lo), float(hi)


def _rgb_to_hsv(rgb):
    from matplotlib.colors import rgb_to_hsv
    return rgb_to_hsv(rgb)


def _hsv_to_rgb(hsv):
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)


def hsv_composite(light, hue, cmap="magma", light_range=None, hue_range=None, light_log=True, hue_log=False,
                  light_pct=(1.0, 99.9), hue_pct=(1.0, 99.0)):
    """uchar RGB (..., 3) and the ranges used: colour from `hue` via `cmap`, scaled in HSV
    value by normalised `light` (vizmo's composite). Missing light -> black; missing hue -> mid colour.

    Ranges default to percentiles of the data (log ranges over positive values only).
    """
    light_range = light_range or auto_range(light, light_log, light_pct) or (1.0, 10.0)
    hue_range = hue_range or auto_range(hue, hue_log, hue_pct) or (0.0, 1.0)
    lt = np.nan_to_num(normalize(light, *light_range, light_log), nan=0.0)
    ht = np.nan_to_num(normalize(hue, *hue_range, hue_log), nan=0.5)
    table = colormap_table(cmap)
    rgb = table[np.round(ht * (len(table) - 1)).astype(int)]
    hsv = _rgb_to_hsv(rgb)
    hsv[..., 2] *= lt
    out = np.round(255.0 * _hsv_to_rgb(hsv)).astype(np.uint8)
    return out, light_range, hue_range


# --------------------------------------------------------------------------- ParaView presets
def preset_json(name, n=64):
    """ParaView colour-map preset (dict) sampled from matplotlib map `name`."""
    t = colormap_table(name, n)
    x = np.linspace(0.0, 1.0, n)
    pts = np.column_stack([x, t]).ravel().round(6).tolist()
    return {"Name": preset_name(name), "ColorSpace": "Lab", "NanColor": [0.0, 0.0, 0.0], "RGBPoints": pts,
            "Creator": "pvgizmo (matplotlib)"}


def register_presets(names=GENERATED, directory=None):
    """Add the matplotlib maps ParaView lacks to its preset list (client side; idempotent).

    Returns the names added. ParaView keeps imported presets in its user settings.
    """
    from paraview import servermanager as sm
    presets = sm.vtkSMTransferFunctionPresets.GetInstance()
    todo = [n for n in names if not presets.HasPreset(preset_name(n))]
    if not todo:
        return []
    import tempfile
    directory = directory or tempfile.gettempdir()
    path = os.path.join(directory, f"pvgizmo_presets_{os.getpid()}.json")
    with open(path, "w") as fh:
        json.dump([preset_json(n) for n in todo], fh)
    try:
        presets.ImportPresets(path)
    finally:
        os.remove(path)
    return [preset_name(n) for n in todo]
