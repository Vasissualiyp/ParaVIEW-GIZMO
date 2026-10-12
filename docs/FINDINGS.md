# Findings so far (2026-10-12)

## Dev environment (`flake.nix`)

`nix develop` gives ParaView 6.0.1 (Qt GUI, Python, MPI) plus a Python env with
meshoid's dependencies, a C++ plugin toolchain (cmake, ninja, gcc, gdb) and HDF5.

- **nixpkgs is pinned** to `151fa4e8…` (the system's NixOS revision). On the
  newest nixos-unstable (`6510408…`), `vtk-9.5.2` and `paraview-6.0.1` were not
  in cache.nixos.org yet, so Nix would have compiled both from source. On the
  pinned revision, nothing heavy builds locally (about 212 MiB download). **When
  you bump nixpkgs, run
  `nix build --dry-run .#devShells.x86_64-linux.default` and confirm that vtk and
  paraview are listed under "fetched", not "built".**
- **ParaView embeds VTK's interpreter** (`vtk-full.vtkPackages.python3Packages`,
  Python 3.14.7). The Python env is built from that package set, so
  numba/h5py/scipy are ABI-compatible with `pvpython` and the GUI. The shellHook
  puts that env's site-packages on `PYTHONPATH`.
- **meshoid comes from the local clone** `./meshoid`
  (github.com/mikegrudic/meshoid @ `ad32966`, v1.52.0, git-ignored). It is put on
  `PYTHONPATH` like an editable install. Nix provides only its dependencies:
  numpy, scipy, numba 0.67, h5py, astropy.
- `PV_PLUGIN_PATH=$PWD/plugin` is exported when `plugin/` exists, so the
  plugins auto-load in the GUI.
- `NUMBA_CACHE_DIR` points to `~/.cache/numba`, so JIT results persist across
  ParaView sessions.
- GUAC's flake (`~/research/software_src/GUAC/flake.nix`) was the starting point.
  It builds meshoid 1.48.5 from PyPI with a custom setup.py install. The
  upstream package now uses `pyproject` and setuptools, so that hack isn't
  needed.

### Verified
| Check | Result |
|---|---|
| `pvpython --version` | 6.0.1 |
| `import meshoid, numba, h5py` inside pvpython | OK, VTK 9.5.2, numba 0.67.0 |
| `Meshoid(...).SurfaceDensity()`, `.Density()` inside pvpython on 2e4 random points | OK |
| Load a `VTKPythonAlgorithmBase` + `@smproxy` Python plugin via `LoadPlugin` | OK |
| `find_package(ParaView)` in a C++ CMake project (configure only, no compile) | OK. The project must enable **C and CXX** (catalyst → FindMPI needs C) |
| **C++ plugin compile + load** (re-verified 2026-10-12): `paraview_plugin_scan/build`, a VTK module with a `vtkPolyDataAlgorithm` + SM XML, built with Ninja, then `LoadPlugin` in pvpython | OK. The export header is named after the module's *short* name (`Mod` → `ModModule.h`, `MOD_EXPORT`). The `paraview.simple` function name comes from the XML `label` with spaces stripped |
| **GUI** under `xvfb-run -a paraview --script=… --exit`: C++ plugin + meshoid + `SaveScreenshot` | OK (no display in the agent session, so Xvfb was used; `QT_QPA_PLATFORM=offscreen` aborts because it has no GL) |
| `vtkSPHCubicKernel().GetCutoffFactor()` | 2.0, so **h_vtk = H_gizmo / 2** for the built-in SPH interpolators |
| VisIt in the pinned nixpkgs | **not packaged** (no `visit` attribute) |
| Built-in SPH tools present | `SPHInterpolator`, `SPHVolumeInterpolator`, `SPHPlaneInterpolator`, `SPHLineInterpolator`, `vtkSPHCubicKernel`/`Quartic`/`Quintic`; Point Gaussian mapper is available |

### Gotchas found
- **`multiprocessing` under pvpython/paraview (root cause found).** numba's
  parallel backend init (`numba/np/ufunc/parallel.py:308`) creates a
  `multiprocessing` RLock, and that starts `resource_tracker` with
  `sys.executable`. That is `pvpython`/`paraview`, not python. In pvpython this
  prints `can't open file '…from multiprocessing.resource_tracker import main…'`.
  In the GUI it gives `NumbaSystemWarning: Could not obtain multiprocessing
  lock … Broken pipe`. Results are correct in both cases. scipy's KDTree
  `workers=` uses threads and is not involved. **Fix (verified):** the shell
  exports `PV_GIZMO_PYTHON`, and every plugin calls
  `multiprocessing.set_executable(os.environ["PV_GIZMO_PYTHON"])` before it
  imports numba or meshoid. With that, the warning goes away.
- The meshoid clone is git-ignored, so it can disappear on a fresh checkout.
  The shellHook now re-clones it at `ad32966` if it is missing.
- No local GIZMO snapshot was found under `~` (depth ≤ 6). A public real
  snapshot is used by meshoid's Walkthrough: STARFORGE
  `M2e4_R10_…_42/output/snapshot_2590.hdf5` (3.1 GB, 23.7M gas cells, no
  cosmology, velocities in m/s), from
  `users.flatironinstitute.org/~mgrudic/starforge_data/STARFORGE_RT/STARFORGE_v1.2/`.
  It has not been downloaded yet.

## Meshoid (read: `src/meshoid/*.py`, `docs/source/*.rst`)

`Meshoid(pos, m=None, kernel_radius=None, des_ngb=32, boxsize=None, particle_mask=None, n_jobs=-1)`.
Neighbours, kernel radius (H, the support radius of the cubic spline, as in
GIZMO), density and volume are computed lazily by `TreeUpdate()`, which uses a
scipy `KDTree` and numba `HsmlIter`.

| Capability | API | Output | Use in ParaView |
|---|---|---|---|
| Kernel radius / density / volume | `SmoothingLength()`, `Density()`, `Volume()` | (N,) | point arrays (e.g. when hsml is missing: stars, DM) |
| Gradients | `D(f, order=1/2)`, `D2(f)`, `Div(v)`, `Curl(v)` | (N,…,3) | derived-field filter |
| Kernel stats | `KernelAverage`, `KernelVariance`, `Integrate` | (N,) / scalar | derived fields / reductions |
| Reconstruction at points | `Reconstruct(f, pts, order 0–2, slope_limiter)` | (M,…) | basis for slices/probes at arbitrary points |
| 2D slice | `Slice(f, size, plane, center, res, order)` | (res,res) | `vtkImageData` slice |
| 3D grid | `InterpToGrid` (reconstruct), `DepositToGrid` (conservative SPH deposit) | (res³) | `vtkImageData` → volume rendering, contours, streamlines |
| Projections | `SurfaceDensity` (conservative / multigrid / **cuda** backend), `Projection`, `ProjectedAverage`, `MassWeightedAverage` | (res,res) | column-density maps |
| Low level | `grid_deposition.GridSurfaceDensity[Multi]`, `Grid_PPZ_DataCube` (PPV cubes), `GridDensity` | | |
| I/O | `load_from_snapshot(value, ptype, sdir, snum)` (PFH's GIZMO HDF5 loader: multi-file, cosmological → physical units) | | reader backend |

Important limits:
- Projections **integrate along z only**, centred on `center` with side
  `size`. For other view directions, rotate the coordinates first; this is cheap.
- Grids are cubic: one `size` and one `res` for all axes.
- Smoothing lengths are clipped to at least about 2 grid cells, so projections
  are resolution-dependent.
- `SurfaceDensity(backend="cuda")` exists and needs `numba-cuda`, which is not
  in the env yet. The machine has an RTX 3070 Ti.
- The GIZMO kernel radius H is the **support radius**. VTK's SPH kernels take a
  smoothing length h with support of about 2h for cubic splines. Using ParaView's
  built-in SPH filters therefore needs a factor-of-2 conversion; check
  `vtkSPHKernel` before relying on it.
