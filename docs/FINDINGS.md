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
- **GUI scripts and the Welcome dialog.** ParaView 6 opens a modal Welcome
  dialog on a timer. It appears in the first event-loop spin of a
  `--script` run and blocks forever under Xvfb. Calls that spin the loop
  include `UpdateAnimationUsingDataTimeSteps()` and setting `AnimationTime`.
  This was found with gdb (`ParaViewMainWindow::showWelcomeDialog` →
  `QDialog::exec`). Fix: an isolated `XDG_CONFIG_HOME` whose `ParaView.ini`
  contains `[General]` `GeneralSettings.ShowWelcomeDialog=false`. The key has
  a dot, so it lives under `[General]`, not `[GeneralSettings]`.
  - In the GUI, `sys.stderr` goes to the output window.
  - `print` needs `flush=True` before `--exit`.
- **Reader output reuse.** VTK hands `RequestData` the same output object
  every time, so it must be cleared with `Initialize()`. Otherwise blocks
  accumulate across time steps or property changes.
- **`ExtractBlock` selector mode depends on timing.** The default `Assembly`
  is `Hierarchy` (`/Root/<block name>`) if the input hasn't executed yet,
  and the dataset's own `vtkDataAssembly` if it has. Naming our assembly
  root `Root` makes `/Root/Gas` valid in both cases.
- The meshoid clone is git-ignored, so it can disappear on a fresh checkout.
  The shellHook now re-clones it at `ad32966` if it is missing.
- **Real test data (2026-10-12):** `~/research/trillium/scratch/COPY/2026-03/`
  is an **sshfs mount of Trillium scratch** (about 10 MB/s). It holds about
  2250 snapshots of FIRE m12f cutouts (cosmological: z≈21, h=0.702,
  BoxSize=60000). Sizes range from bytes to 41 GB, with a median of 390 MB.
  Small ones for tests: `m12f_cutout/output_cutout/snapshot_259.hdf5`
  (31 MB, 74k gas cells), `…/snapshot_519.hdf5` (184 MB, 442k gas cells),
  `m12f_cutout/output_jeans_refinement/snapshot_177.hdf5` (234 MB, 561k gas
  cells plus DM).
  - Gas has 28 fields, including `EddingtonTensor` (6 components),
    `PhotonEnergy` (5 components) and `Metallicity`. PartType5 holds
    sinks/BHs with STARFORGE-style fields.
  - In the `output_cutout` files the **header says 1737 PartType1 particles,
    but there's no `PartType1` group**, so we trust the groups.
  - Every field matches `load_from_snapshot` in both unit modes. The only
    differences are float32 denormals, because we keep float32 while PFH
    upcasts to float64.
  - Reading the 7 default arrays of the 561k-cell file takes about 5 s, and
    that's sshfs speed.
  - **Float32 rendering collapse.** In `snapshot_259` the whole gas region
    is **0.064 pc across at about 1948 kpc** from the origin. Coordinates are
    float64 on disk, but rendering is float32, whose spacing there is 0.12 pc,
    so all 74k cells landed on a single pixel. Fix: the reader recentres in
    float64 before VTK when asked (`Recenter`, opt-in: "Gas density peak" or
    "Custom center"; offset in the `CenterOffset` field array). With
    `Recenter` = None the reader warns when the data would collapse. Meshoid filters must also work in the
    recentred frame, or in float64 absolute coordinates.
- Earlier: no local GIZMO snapshot was found under `~` (depth ≤ 6). A public real
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

## Phase 2 findings (2026-10-12)
- **meshoid's deposition grid is a node grid.** In non-periodic mode every
  deposition routine (2D/3D, CPU/CUDA) puts values at
  `-size/2 + k·size/(res-1)`, corner to corner. Reconstruction (`Slice`,
  `InterpToGrid`) uses cell centres instead, so meshoid's slices and
  projections are offset by up to half a pixel and scaled by res/(res-1).
  "Conservative" deposition conserves mass exactly only with pixel area
  `(size/(res-1))²`; with `(size/res)²` it looks about 3% low at res=64.
  Our filters put **both** slices and projections on the node grid, so they
  overlay exactly (`pvgizmo.meshoid_ops.grid_coords`, `pixel_area`). This is
  worth reporting upstream: the docs don't say it.
- **Projections need no KD-tree** when H is known (gas). Every
  `SurfaceDensity`/`Projection`/`MassWeightedAverage`/`ProjectedAverage` is
  a thin wrapper around `grid_deposition.GridSurfaceDensity[Multi]` /
  `GridAverage`. We call these directly on coordinates rotated into the view
  frame, so changing orientation is just a rotation. Volumes for the
  line-of-sight integral are `m/Density` from the snapshot (GIZMO's own),
  falling back to meshoid's `4π/3 H³/32`.
- **Speed:** 561k gas cells at 256² take 0.06 s (numba-threaded, JIT cached
  in `NUMBA_CACHE_DIR`), so maps update interactively. Slices build a
  KD-tree once, cached per block in an LRU of 2.
- **Real-data acceptance:** the plugin map equals GUAC's
  `Meshoid(pos, m, hsml).SurfaceDensity(M.m, center, size, res)` on m12f
  snapshots 259 and 177 to about 1e-8 relative (float64 rounding at a
  1948 kpc offset).
- **GUAC `scripts/gal_viz.py` plots transposed maps.** It calls
  `np.meshgrid(X, Y)` (default `xy` indexing) then `pcolormesh(X, Y, sigma)`,
  but meshoid's first index is x. Other GUAC scripts use `indexing='ij'`,
  which is correct. Not changed here, since it's the user's repo.
- **Array-selection defaults need an executed input.** A filter's `Field`
  default (`Masses`/`Density`) only applies if the input has already run
  when the filter is created. The GUI always does this; in scripts call
  `UpdatePipeline()` on the reader first (`pvhelpers.open_snapshot` does).
  Multi-element string defaults need `default_values_delimiter=";"`.
- **Python-plugin decorators** take the number of elements from the method
  signature, so pass `number_of_elements` explicitly when wrapping. Class
  decorators that add properties must be innermost, below `@smproxy.filter`.
- **Colour maps:** collapsing clouds span about 8 decades in column density.
  A fixed "max/1e4" log range gives a black image, so `pvhelpers.show_map`
  uses the 1st–99.9th percentiles of positive pixels.
- **Live view-dependent projections (verified with real mouse drags).** In the
  GUI, `pvgizmo.live` keeps FollowCamera projections and slices aimed at the
  camera: centre = focal point, normal toward the camera, size = visible
  extent at the focal plane (covering the wider viewport side). While dragging
  they run at 128²; on release they run once at full resolution. The xdotool
  test saw 20 projections during a 10-step drag, then one full-resolution
  run. Mouse start/end events arrive on the view proxy.

## Phase 3 findings (2026-10-12)
- **meshoid 3D deposition, three problems on zoom data**
  (`grid_deposition.GridDensity`, used by `Meshoid.DepositToGrid`):
  1. Its non-periodic grid sits **half a voxel off** its own 2D grid: node k is
     at `-size/2 + (k+1/2)·dx`. In 2D it is at `-size/2 + k·dx`.
  2. Each particle's weights are **renormalised over the in-box nodes only**,
     so a kernel crossing the box edge deposits its *whole* mass inside. On
     m12f snapshot 177, a window around the density peak overlaps 4.3k
     edge-crossing cells, 157 of them with kernels larger than the whole box.
     Together they carry **440× the mass actually inside the window**.
  3. It allocates a (2H/dx)³ kernel cache per particle. With H/dx up to
     77,000 that is a MemoryError, which the threaded wrapper then hides as an
     unrelated `out += None` dtype error.

  The 2D *conservative* core has the same renormalisation and cache. The
  default non-conservative 2D core (what GUAC uses) is fine: it uses an
  analytic `f·W/h²`. **Fix:** `pvgizmo/deposit.py`, a numba kernel that
  normalises over the *full* kernel (exact discrete sum up to 15 voxels,
  analytic beyond), visits only in-box nodes and allocates nothing. It
  matches meshoid exactly for kernels inside the box. A 128³ grid of the
  561k-cell snapshot takes 0.42 s. Conservative 2D maps send edge-crossing
  kernels through the non-renormalised core. Worth reporting upstream.
- **`TreeUpdate` overwrites a user-supplied kernel radius.** Accessing
  `ngb` (gradients, order ≥ 1 reconstruction) recomputes `kernel_radius`,
  `density` and `vol` with meshoid's own 32-neighbour estimate. The Derived
  Fields filter therefore names its outputs `Meshoid*`.
- `Meshoid.KernelAverage` is scalar-only (`einsum "ij,ij->i"`), so we apply it
  per component. `KernelVariance` returns the standard deviation.
- **ParaView's volume mapper ignores log-scaled colour maps** (a black
  render). Meshoid Grid therefore also outputs `log10_[array]`, which
  `pvhelpers.show_volume` renders with a linear map. Opacity starts higher
  than the colour range, so visible voxels get colour.
- **A `<` in a Python plugin's XML (e.g. a Documentation string) makes that
  proxy silently fail to register.** `test_all_proxies_registered` guards
  against this.
- Array names are Calculator-friendly (`SurfaceDensity_Masses`,
  `grad_Density`, `log10_GridDensity_Masses`). Names with parentheses need
  quoting in ParaView's Calculator.
- Timings (561k gas cells, cached tree): gradient + div + curl 5.8 s the
  first time (tree and weights), then kernel average 3.4 s; velocity
  interpolation onto a 64³ grid 0.17 s.

## Phase 4 findings (2026-10-12): performance and client-server
- **Benchmark** (`bench/bench.py`, synthetic Plummer + background with analytic
  H, 1M/4M/16M gas, local disk). Before optimising, a camera move at 128²
  cost 0.04 / 0.15 / 0.60 s, nearly independent of zoom. At 16M it split into
  0.27 s float32→float64 cast + rotation, 0.05 s kernel-radius cast/clip, and
  only 0.15 s deposition.
- **Subsampling while dragging (LOD) does not help.** With 2M of 16M
  particles and H scaled by (N/n)^(1/3), it took 0.23 s against 0.15 s for
  the full set, because the bigger kernels cover more pixels. Dropped.
- **Fix:** `pvgizmo.deposit.project2d`, a numba kernel that rotates, culls and
  clips in-loop, reading float32 arrays in place. It is identical to meshoid's
  `GridSurfaceDensity` to 7e-16. View-independent per-particle channels
  (m·f, volumes, the automatic window) are cached per dataset in the filter.
  **One interactive camera move (filter re-execution, 128²) now costs
  0.033 s (1M), 0.055 s (4M) and 0.142 s (16M)**, about 30 / 18 / 7 frames/s.
  The full-resolution frame after release at 512² takes 0.3 / 0.5 / 0.84 s.
- Slices: the first call builds meshoid's KD-tree (0.6 s at 1M, 3 s at 4M,
  12–14 s at 16M); re-aiming afterwards costs 0.01–0.07 s.
  `balanced_tree=False` only saves 20%, so it isn't used. Derived Fields
  gradients: 4 s at 1M, 17 s at 4M (meshoid neighbour search and weights).
- **ParaView's client-server delivery drops `vtkImageData`'s direction
  matrix.** A rotated map rendered on the client arrives axis-aligned. A plane
  seen edge-on showed 4.8% visible pixels with local rendering, versus 0% in
  builtin mode and with server-side rendering. 2D maps are therefore
  `vtkStructuredGrid` with explicit points. 3D grids are axis-aligned and stay
  `vtkImageData`. `vtkStructuredGrid.GetDimensions()` needs an argument in
  VTK 9.5's Python, so use `GetExtent()`.
- **Client-server works end to end** (`tests/test_client_server.py`, and the
  GUI drag test against a local `pvserver`). Plugins load on both sides via
  `PV_PLUGIN_PATH`, the server opens the files, the camera link runs on the
  client and the filter on the server. The FollowCamera auto-linker is also
  installed at plugin import, because in client-server the filter
  constructor runs on the server. `pvserver` exits when its client
  disconnects.
- **GPU (RTX 3070 Ti) is untested:** `nvidia-smi` reports "Driver/library
  version mismatch" (the driver was updated without a reboot). `numba-cuda`
  is available in the pinned nixpkgs. A CUDA port of `project2d` would mainly
  help full-resolution frames at large N.
- **Cluster image (verified locally):** `nix build .#pvserver-image` produces a
  1.6 GB docker archive. Apptainer needs it gunzipped, then makes a 1.55 GB
  SIF. Running it with `--containall` (no host /nix, GPU or X) works:
  - the host's ParaView 6.0.1 client connects;
  - server-side rendering works through Mesa EGL, after a harmless "bad X
    server connection" warning;
  - the full client-server test passes.

  See docs/TRILLIUM.md.
- **MPI pvserver duplicated data.** Every rank executed the reader, and
  ParaView merged the pieces: 2 ranks gave 13,000 particles for 6,500. Now
  the reader loads only on piece 0, and filters return empty outputs on empty
  pieces of a parallel run. `mpiexec -n 3 pvserver` gives the single-process
  results (`test_mpi_pvserver`). Real distributed processing (each rank
  reading a slab, images summed with an MPI reduction, neighbour ghosts for
  slices and gradients) is future work. One process already uses every core
  via numba threads.

