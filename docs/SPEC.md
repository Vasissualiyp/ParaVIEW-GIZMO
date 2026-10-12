# Spec: GIZMO data in ParaView via Meshoid (draft v1)

## Goal
Open GIZMO HDF5 snapshots directly in ParaView, interactively. Particles must
be treated as a **meshless volume** (kernel-weighted), not as plain points, so
slices, projections, volume renders, contours and derived fields come out
physically correct. Meshoid supplies the numerics: kernel radii, densities,
gradients, reconstruction and deposition.

## Approach
**Phase 1 is pure Python**, written as ParaView Python plugins
(`VTKPythonAlgorithmBase` + `paraview.util.vtkAlgorithm` decorators). The
pieces are:
- a thin, ParaView-independent core package (`pvgizmo/`) for I/O, units and
  numpy↔VTK conversion
- plugin files that wrap the core as a reader and filters

A C++ port is deferred until profiling shows which parts are actually too slow
(see Phase 4).

### Proposed layout
```
flake.nix
pvgizmo/            # pure python, unit-testable without ParaView
  io.py             # snapshot discovery, multi-part files, header, unit conversion
  meshoid_ops.py    # cached Meshoid construction + projection/slice/grid helpers
  vtkconv.py        # numpy -> vtkPolyData / vtkImageData (zero-copy via numpy_support)
plugin/
  GIZMOReader.py    # reader proxy
  MeshoidFilters.py # filter proxies
tests/
docs/
```

## Components

### 1. `GIZMO Snapshot Reader`
- **Files:** `snapshot_NNN.hdf5`, and `snapdir_NNN/snapshot_NNN.k.hdf5` for
  multi-part snapshots, picked by file extension. A series of snapshots becomes
  a ParaView time series: `TIME_STEPS` carry the header `Time`, with redshift
  shown when the run is cosmological.
- **Output:** `vtkPartitionedDataSetCollection`, with one partition per
  `PartType{0..5}`. Each partition is a `vtkPolyData` of vertex cells, and the
  block names are human-readable (Gas, DM, Disk, Bulge, Stars, BH). This lets
  users pick blocks and colour each particle type separately.
- **Properties:**
  - selection of particle types
  - selection of point arrays, so only the requested HDF5 datasets are read
  - stride/subsample factor and an optional bounding-box crop, for big snapshots
  - physical vs. code units: cosmological a and h handling as in
    `load_from_snapshot`
  - optional "compute kernel radius" for types without `SmoothingLength`, via
    meshoid
- **Arrays:** exposed as point data. Vectors (velocities, B) are 3-component.
  Wide fields such as `Metallicity[:, k]` are split or named per component.
  `ParticleIDs` is kept for tracking. Field data carries header metadata
  (BoxSize, redshift, units).

### 2. `Meshoid Projection` (filter → 2D `vtkImageData`)
- **Modes:**
  - surface density (`SurfaceDensity`, with conservative and multigrid options)
  - line-of-sight integral (`Projection`)
  - mass-weighted average (`MassWeightedAverage`)
  - projected average
- **Inputs:** the field to project, the weight field, centre, size, resolution,
  `smooth_fac`, and backend (cpu, or cuda if available).
- **View direction:** set by a normal vector or by "use current camera". The
  coordinates are rotated so that the view axis maps to z before calling
  meshoid.
- **Output:** a 2D image placed in 3D at the projection plane, so it overlays
  with particles and slices. Log-scaling is left to the colour map.

### 3. `Meshoid Slice` (filter → 2D `vtkImageData`)
- Uses `Reconstruct` at plane points: arbitrary normal and origin, order 0/1/2,
  optional slope limiter, with the planar `Slice` as the fast path.
- Stretch goal: an interactive plane widget, the same UX as ParaView's Slice.

### 4. `Meshoid Grid` (filter → 3D `vtkImageData`)
- Two modes: conservative deposit (`DepositToGrid`, for densities of conserved
  quantities) or interpolation (`InterpToGrid`, for intensive fields).
- Options: resolution and region (box widget).
- The output feeds ParaView's own volume rendering, Contour, Stream Tracer and
  Clip.

### 5. `Meshoid Derived Fields` (filter, particles → particles)
- Adds point arrays: kernel radius, density, volume, gradient, divergence and
  curl of a selected array, kernel average and kernel variance.

### 6. Presets / convenience
- A Python macro that loads a snapshot, picks the gas block, shows it with a
  Point Gaussian representation (scale = kernel radius), adds a surface-density
  projection, and sets a log colour map.

## Cross-cutting requirements
- **Caching:** cache the Meshoid object (tree, neighbours, gradient weights)
  per (file, part type, mask) in the pipeline, so changing projection
  parameters doesn't rebuild the KD-tree.
- **Responsiveness:**
  - default resolution of 256
  - first-call numba JIT warm-up, using the persistent `NUMBA_CACHE_DIR`
  - long operations report progress via `UpdateProgress`
- **multiprocessing:** fix the `sys.executable == pvpython` problem described
  in FINDINGS.
- **Units:** always record the unit system in field data. Physical units are
  the default.
- **Tests:** pytest for `pvgizmo/`, using a synthetic snapshot written with
  h5py (uniform box, Plummer sphere) and comparing against meshoid called
  directly. A pvpython smoke test loads the plugins and runs each filter.

## Key design decisions (v1)
1. **Data model.** The reader emits a `vtkPartitionedDataSetCollection` plus a
   `vtkDataAssembly` (`/Gas`, `/Stars`, …). Particles are `vtkPolyData` with a
   single poly-vertex cell, which is cheap and renders with Point Gaussian. The
   kernel radius goes in a point array named `KernelRadius` (H, the support
   radius), so downstream filters never guess.
2. **Meshoid filters take particles and return grids.** Every Meshoid filter
   takes the reader's collection, a `Block` selector (default Gas) and an input
   array. Output is `vtkImageData` with origin and spacing in simulation
   coordinates, so it overlays exactly with the particles. A 2D image is one
   cell thick, placed at the projection or slice plane.
3. **Arbitrary orientation.** For orientation, build an orthonormal basis
   (u, v, n) from `Normal` + `ViewUp`, then rotate positions and vectors into
   it and call meshoid's z-only routines. The output image gets a VTK
   direction matrix (`vtkImageData.SetDirectionMatrix`), so it appears
   correctly oriented in 3D. "Project along current camera" is a client-side
   Python macro that copies the camera's normal and view-up into the filter
   properties, because a server-side algorithm can't see the view.
4. **Meshoid cache.** A module-level LRU (size 2) keyed on (input data object
   id + MTime, block, particle-mask hash) holds the `Meshoid` instance.
   Changing resolution, field or size reuses the tree, neighbours and
   gradient weights. Rotations reuse it too, because rotation is applied to a
   copy of `pos` passed into the grid routines, not to the tree.
   *Caveat:* `SurfaceDensity` and the related routines read `self.pos`.
   Rotation is therefore done either with a lightweight `Meshoid` that shares
   `kernel_radius` (skipping `TreeUpdate`), or by calling `grid_deposition.*`
   directly with `(f, x_rot, h)`. Direct calls are preferred, since they are
   the documented low-level API.
5. **Built-in VTK SPH filters stay usable.** The reader also writes
   `SmoothingLengthVTK = H/2`, so ParaView's own `SPHVolumeInterpolator` and
   `SPHPlaneInterpolator` (cubic, cutoff factor 2) give comparable results.
   These are useful as a cross-check and for line probes.
6. **Process hygiene.** Each plugin file starts with
   `multiprocessing.set_executable(os.environ.get("PV_GIZMO_PYTHON", ...))`
   and only then imports meshoid. numba JIT results are cached in
   `NUMBA_CACHE_DIR`. A `Warm up JIT` step runs on plugin load (a tiny
   2000-particle call) so the first real projection isn't slow.
7. **Time series.** `RequestInformation` opens only the HDF5 headers of every
   matched snapshot to build `TIME_STEPS`, and lazy-loads the requested step
   in `RequestData`. Datasets are read with `h5py` hyperslabs when a
   stride or crop is set.
8. **Python first, C++ later.** The C++ plugin toolchain is verified (see
   FINDINGS), so a native port of the reader or deposition is an option in
   Phase 4, not a blocker. The Python plugins and a later C++ plugin can share
   the same XML/proxy names, so state files keep working.

## Phase 1–2 work breakdown
| # | Task | Output |
|---|---|---|
| 1.1 | `tests/make_snapshots.py`: write synthetic GIZMO HDF5 with h5py (uniform box, Plummer sphere, multi-part `snapdir`, 3-step series, cosmological header) | fixtures |
| 1.2 | `pvgizmo/io.py`: discovery, header, field list, hyperslab reads, units/cosmology (cross-checked against `meshoid.load_from_snapshot`) | pytest green |
| 1.3 | `pvgizmo/vtkconv.py`: numpy → polydata / image (zero-copy, keep refs alive) | pytest green |
| 1.4 | `plugin/GIZMOReader.py`: proxy with type/array selection, stride, units toggle, time steps | opens fixtures in pvpython + GUI (xvfb screenshot test) |
| 2.1 | `pvgizmo/meshoid_ops.py`: cache, rotation basis, projection/slice wrappers | matches direct meshoid to 1e-12 |
| 2.2 | `plugin/MeshoidFilters.py`: Projection (4 modes) and Slice | smoke tests; visual check against GUAC `center_and_project.py` on STARFORGE data |
| 2.3 | Macro "Project along camera" + Point Gaussian preset | manual GUI check |

## Phases
| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Nix dev shell (done) | pvpython + GUI import meshoid; Python **and C++** plugins build and load; multiprocessing fix |
| 1 | Reader + synthetic test snapshots | single- and multi-part snapshots and time series open in the GUI, with arrays selectable |
| 2 | Projection + Slice filters | maps match meshoid/GUAC scripts on the same data |
| 3 | Grid + Derived Fields; Point Gaussian preset | volume rendering and contours of gas density work |
| 4 | Performance | profile real FIRE/STARFORGE data. Options: C++ reader (HDF5) and/or C++ deposition; MPI-parallel reading split by file parts; `pvserver` on a cluster with client–server mode; CUDA backend (`numba-cuda`) |
| 5 | Packaging | Nix package for the plugin; documented usage; optional VisIt route (not yet investigated) |

## Open questions for you
1. Test data: which snapshots, and where? (FIRE, STARFORGE, idealised boxes;
   typical particle counts.) May I download the public 3.1 GB STARFORGE
   walkthrough snapshot as the real-data benchmark?
2. Is VisIt support wanted too, or is ParaView enough for now? VisIt is not
   in nixpkgs, so it would need its own (heavy) packaging or a non-Nix
   install. VisIt can read the 3D grids we export (`.vti`/`.vtk`) without any
   plugin.
3. Do you need remote/cluster use (`pvserver` on Trillium) early, or is local
   use enough?
4. Should the plugin eventually be upstreamed into meshoid, or stay as a
   separate repo?
