"""GIZMO HDF5 snapshot I/O, independent of ParaView.

Conventions follow PFH's ``load_from_snapshot`` (shipped with meshoid):
multi-part snapshots are ``<base>.<k>.hdf5`` (optionally inside ``snapdir_NNN/``),
particle groups are ``PartType0..5``, and the "physical units" conversion
applies the a/h factors for the fields it knows about.
"""

import os
import re

import h5py
import numpy as np

NTYPES = 6
PTYPE_NAMES = ("Gas", "DarkMatter", "Type2", "Type3", "Stars", "Type5")

_PART_RE = re.compile(r"^(?P<base>.*?)(?:\.(?P<part>\d+))?(?P<ext>\.hdf5|\.h5)$")
_SNUM_RE = re.compile(r"_(\d+)(?:\.\d+)?\.(?:hdf5|h5)$")


def snapshot_parts(path):
    """Return the ordered list of files making up the snapshot that contains `path`.

    `path` may be a single-file snapshot, any part ``<base>.<k>.hdf5`` of a
    multi-part snapshot, or a ``snapdir_NNN`` directory.
    """
    if os.path.isdir(path):
        files = sorted(
            (f for f in os.listdir(path) if _PART_RE.match(f)),
            key=lambda f: int(_PART_RE.match(f).group("part") or 0),
        )
        if not files:
            raise FileNotFoundError(f"no HDF5 snapshot files in {path}")
        path = os.path.join(path, files[0])
    m = _PART_RE.match(path)
    if m is None:
        raise ValueError(f"not an HDF5 snapshot file name: {path}")
    with h5py.File(path, "r") as f:
        nfiles = int(f["Header"].attrs.get("NumFilesPerSnapshot", 1))
    if nfiles <= 1 or m.group("part") is None:
        return [path]
    return [f"{m.group('base')}.{k}{m.group('ext')}" for k in range(nfiles)]


def snapshot_key(path):
    """A key identifying the snapshot a file belongs to (same for all its parts)."""
    m = _PART_RE.match(path)
    return m.group("base") if m else path


def snapshot_number(path):
    m = _SNUM_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def read_header(parts):
    """Header attributes of the first part, as plain Python/numpy values."""
    with h5py.File(parts[0], "r") as f:
        hdr = {k: (v.item() if np.ndim(v) == 0 and hasattr(v, "item") else v)
               for k, v in f["Header"].attrs.items()}
    hdr["NumPart_Total"] = total_counts(hdr)
    return hdr


def total_counts(hdr):
    lo = np.asarray(hdr.get("NumPart_Total", np.zeros(NTYPES)), dtype=np.int64)
    hi = np.asarray(hdr.get("NumPart_Total_HighWord", np.zeros(NTYPES)), dtype=np.int64)
    return lo + (hi << 32)


def is_cosmological(hdr):
    """Same heuristic as load_from_snapshot: Time == a == 1/(1+z)."""
    if "ComovingIntegrationOn" in hdr:
        return bool(hdr["ComovingIntegrationOn"])
    return abs(hdr["Time"] * (1.0 + hdr["Redshift"]) - 1.0) < 1e-6


def physical_factor(name, hdr):
    """Multiplicative factor converting field `name` from code to physical units.

    Mirrors load_from_snapshot exactly (including its non-cosmological h
    handling), so results agree with GUAC/meshoid scripts.
    """
    h = float(hdr.get("HubbleParam", 1.0))
    a = float(hdr["Time"]) if is_cosmological(hdr) else 1.0
    hinv = 1.0 / h
    rconv = a * hinv
    if name in ("Coordinates", "SmoothingLength"):
        return rconv
    if name == "Velocities":
        return np.sqrt(a)
    if name in ("Density", "Pressure"):
        return hinv / rconv**3
    if name == "StellarFormationTime" and not is_cosmological(hdr):
        return hinv
    if name in ("Masses", "CosmicRayEnergy", "PhotonEnergy") or "BH_Mass" in name:
        return hinv
    return 1.0


def list_fields(parts):
    """{ptype: {field: (trailing_shape, dtype)}} over all parts; types with no particles omitted.

    'Masses' is listed for types that only have a MassTable entry.
    """
    out = {}
    for p in parts:
        with h5py.File(p, "r") as f:
            mtab = f["Header"].attrs.get("MassTable", np.zeros(NTYPES))
            for t in range(NTYPES):
                g = f.get(f"PartType{t}")
                if g is None or "Coordinates" not in g or g["Coordinates"].shape[0] == 0:
                    continue
                d = out.setdefault(t, {})
                for k, ds in g.items():
                    if isinstance(ds, h5py.Dataset):
                        d.setdefault(k, (ds.shape[1:], ds.dtype))
                if "Masses" not in d and mtab[t] > 0:
                    d["Masses"] = ((), np.dtype(np.float64))
    return {t: out[t] for t in sorted(out)}


def _part_counts(parts, ptype):
    n = []
    for p in parts:
        with h5py.File(p, "r") as f:
            g = f.get(f"PartType{ptype}")
            n.append(0 if g is None or "Coordinates" not in g else g["Coordinates"].shape[0])
    return np.array(n, dtype=np.int64)


def read_ptype(parts, ptype, fields, physical=True, stride=1, bbox=None, center=None):
    """Read `fields` of one particle type from all parts of a snapshot.

    stride: keep every `stride`-th particle *globally* (consistent across parts).
    center: subtracted from Coordinates (in float64, after the unit conversion).
            Zoom/cutout regions can be far smaller than float32 spacing at their
            absolute position, and rendering is float32: recentre before VTK.
    bbox:   (xmin, xmax, ymin, ymax, zmin, zmax) in the *output* units and frame
            (i.e. after unit conversion and recentring). Coordinates are always read.
    Returns a dict name -> array (empty dict if the type has no particles).
    """
    hdr = read_header(parts)
    counts = _part_counts(parts, ptype)
    if counts.sum() == 0:
        return {}
    fields = ["Coordinates"] + [f for f in fields if f != "Coordinates"]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = {k: [] for k in fields}
    for p, n, off in zip(parts, counts, offsets):
        if n == 0:
            continue
        first = (-off) % stride  # first local index on the global stride grid
        sl = slice(first, None, stride)
        with h5py.File(p, "r") as f:
            g = f[f"PartType{ptype}"]
            mtab = f["Header"].attrs.get("MassTable", np.zeros(NTYPES))
            for k in fields:
                if k in g:
                    out[k].append(g[k][sl])
                elif k == "Masses" and mtab[ptype] > 0:
                    out[k].append(np.full(len(range(n)[sl]), mtab[ptype]))
                else:
                    raise KeyError(f"{k} not present for PartType{ptype} in {p}")
    data = {k: np.concatenate(v) for k, v in out.items()}
    if physical:
        for k in data:
            fac = physical_factor(k, hdr)
            if fac != 1.0:
                arr = data[k]
                if np.issubdtype(arr.dtype, np.floating):
                    arr *= arr.dtype.type(fac)
                else:
                    data[k] = arr * fac
    if center is not None:
        data["Coordinates"] = data["Coordinates"].astype(np.float64) - np.asarray(center, np.float64)
    if bbox is not None:
        x = data["Coordinates"]
        b = np.asarray(bbox, dtype=np.float64).reshape(3, 2)
        mask = np.all((x >= b[:, 0]) & (x <= b[:, 1]), axis=1)
        data = {k: v[mask] for k, v in data.items()}
    return data


def density_peak(parts, physical=True):
    """Position of the densest gas cell (float64), or None if there is no gas density."""
    fields = list_fields(parts).get(0, {})
    if "Density" not in fields:
        return None
    d = read_ptype(parts, 0, ["Density"], physical=physical)
    return d["Coordinates"][np.argmax(d["Density"])].astype(np.float64)
