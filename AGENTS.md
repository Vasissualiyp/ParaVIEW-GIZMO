# AGENTS.md

Guidance for coding agents (and humans) working in this repo.

## What this is
ParaView plugins that treat GIZMO (meshless MFM/MFV, SPH) snapshots as a
kernel-weighted volume rather than plain points, using
[meshoid](https://github.com/mikegrudic/meshoid) for the numerics. Read
`docs/SPEC.md` for the plan and phases, and `docs/FINDINGS.md` for the verified
environment facts and gotchas. Keep both up to date when you learn something
non-obvious.

## Layout
```
flake.nix            nix dev shell: ParaView 6.0.1 + meshoid deps + C++ toolchain + xvfb-run
pvgizmo/             pure-Python core, no ParaView dependency except vtkconv (VTK only)
  io.py              snapshot discovery, headers, field lists, strided/cropped reads, units
  vtkconv.py         numpy -> vtkPolyData / vtkImageData (zero-copy)
  meshoid_ops.py     view bases, projections/slices at any orientation, 3D grids, derived
                     fields, Meshoid LRU cache
  deposit.py         numba kernels: project2d (per-frame projection: in-loop rotation/culling,
                     float32 in place) and deposit3d (3D, correct for box-crossing kernels)
bench/bench.py       Phase 4 benchmark (synthetic 1M-16M snapshots)
  pvhelpers.py       client-side: open_snapshot, show_particles, add_projection,
                     project_along_camera, show_map (needs paraview.simple)
  live.py            client-side camera links: maps follow the view (rotate/zoom/pan), low-res
                     while dragging; auto-installed in the GUI for FollowCamera=1
plugin/              ParaView Python plugins (auto-loaded via PV_PLUGIN_PATH in the shell)
  GIZMOReader.py     "GIZMO Snapshot Reader" -> vtkPartitionedDataSetCollection
  MeshoidFilters.py  Filters > GIZMO: "Meshoid Projection", "Meshoid Slice" (2D vtkImageData),
                     "Meshoid Grid" (3D vtkImageData), "Meshoid Derived Fields" (particles)
macros/              ParaView macros (Macros > Import new macro): ProjectAlongCamera,
                     ToggleFollowCamera, GIZMOQuickLook
tests/               pytest; snapgen.py writes synthetic GIZMO snapshots
meshoid/             git-ignored local clone (pinned; recreated by the shellHook)
```

## Environment
- Always work inside `nix develop` (or `nix develop -c <cmd>`). Don't install
  anything with pip. Add Python dependencies to `pythonEnv` in `flake.nix`.
  They must come from `vtk-full.vtkPackages.python3Packages`, because that is
  ParaView's interpreter ABI.
- nixpkgs is pinned so that VTK and ParaView come from the binary cache. If you
  bump it, run `nix build --dry-run .#devShells.x86_64-linux.default` and make
  sure vtk and paraview are *fetched*, not *built*. Building them from source
  takes hours.
- meshoid is used as an editable clone at `./meshoid`, on `PYTHONPATH`. Change
  meshoid there and upstream the changes. Don't vendor them here.

## Tests
Run everything with ParaView's interpreter:
```sh
nix develop -c pvpython -m pytest -q tests -p no:cacheprovider
```
- `test_io.py` / `test_vtkconv.py` cover the core. `test_io.py` checks
  against meshoid's `load_from_snapshot`, the reference loader.
- `test_meshoid_ops.py` checks every projection mode against meshoid's own
  methods, rotated views against permuted data, exact mass conservation,
  and slice geometry.
- `test_reader_pv.py` / `test_filters_pv.py` / `test_live_pv.py` cover the
  plugins and camera links through the server manager.
- `test_client_server.py` runs `tests/cs_client.py` against a real local
  `pvserver`. It checks plugin loading on both sides, camera-following, and
  rotated maps keeping their orientation through client delivery.
- `test_client_server.py::test_mpi_pvserver` runs `mpiexec -n 3 pvserver`.
  Results must equal the single-process ones: the reader loads on piece 0
  only, and filters return empty outputs on empty pieces.
- Cluster image: `nix build .#pvserver-image` (docs/TRILLIUM.md). It bakes in
  `pvgizmo/` and `plugin/`, so rebuild it after changing them. Test it with
  `apptainer run --containall` plus `tests/cs_client.py`.
- Performance: `nix develop -c pvpython bench/bench.py [N ...]` (writes
  `testdata/bench_*.hdf5`). Re-run it after touching `pvgizmo/deposit.py` or
  the filters, and update the FINDINGS numbers.
- `test_gui.py::test_live_mouse_drag` (builtin and client_server) starts the GUI with `tests/gui_live.py`
  (no `--exit`, so the Qt loop runs) and **drags the real mouse with xdotool**
  (`tests/gui_drag.sh`). It asserts on the JSON events that `pvgizmo.live`
  and the filter write to `$PV_GIZMO_LIVE_LOG`. This is the interactivity
  regression test: keep it green.
- `test_gui.py` runs the real Qt client under `xvfb-run`
  (`tests/gui_smoke.py`). It takes about 10 s. Set `PV_GIZMO_KEEP_SHOT=x.png`
  to keep the screenshot.
- `test_realdata.py` is opt-in and runs on real snapshots:
  ```sh
  D=~/research/trillium/scratch/COPY/2026-03/m12f_cutout
  PV_GIZMO_REAL_SNAPSHOTS="$D/output_cutout/snapshot_259.hdf5:$D/output_jeans_refinement/snapshot_177.hdf5" \
    nix develop -c pvpython -m pytest -q tests/test_realdata.py
  ```
  The COPY tree is an **sshfs mount of Trillium scratch** and reads at about
  10 MB/s. Use files under about 250 MB; many snapshots there are 1–40 GB.
  Never copy snapshots into the repo.
- New features need a synthetic-data test. Extend `tests/snapgen.py` rather
  than adding binary fixtures.

## Conventions that matter
- **Kernel radius:** GIZMO's `SmoothingLength` is the kernel *support* radius
  H. VTK's SPH kernels use h with support 2h, so the reader also emits
  `SmoothingLengthVTK = H/2`. Never mix the two up.
- **Units:** physical-units conversion copies `load_from_snapshot` exactly (a,
  h factors). Fields keep their on-disk dtype (usually float32). Results
  differ from PFH's float64 path only for float32 denormals.
- **Block naming:** one partition per particle type, named `Gas`,
  `DarkMatter`, `Type2`, `Type3`, `Stars`, `Type5`, under an assembly whose
  root is named **`Root`**. Selectors like `/Root/Gas` must keep working in both
  of ParaView's selection modes (see the gotchas below).
- **Recentring (opt-in, default None):** when asked, the reader subtracts a
  centre in float64 (gas density peak or custom) and stores it as
  `CenterOffset`. With None it warns if the data would collapse in float32. Zoom regions can be smaller
  than float32 spacing at their absolute position, and rendering is float32.
  Anything new must work in this recentred frame. Don't cast absolute
  coordinates to float32.
- **Array names** are Calculator-friendly: `<Mode>_<Field>`, `Slice_<Field>`,
  `GridDensity_<Field>`, `Interp_<Field>`, `grad_/div_/curl_<Field>`, and
  `log10_` prefixes. No parentheses.
- **Never deposit with meshoid's `GridDensity` / conservative 2D core on
  windows that cut kernels.** They renormalise onto in-box nodes (see
  FINDINGS). Use `pvgizmo.deposit` / `meshoid_ops.grid3d_deposit`. 3D nodes
  coincide with the 2D map nodes.
- **2D maps are `vtkStructuredGrid`** (explicit points), *not* `vtkImageData` with
  a direction matrix: ParaView's client-server delivery drops the direction
  matrix. 3D grids are axis-aligned `vtkImageData`.
- **Per-frame cost:** keep everything view-independent out of the per-camera-move
  path (cache it in `_VIEW_CACHE` keyed on VTK array address + MTime), and never
  cast or copy the particle arrays per frame.
- **Image grids:** projections and slices use meshoid's non-periodic
  *node* grid, `linspace(-size/2, size/2, res)` along the image axes u and v.
  The view basis is right-handed (u, v, n), with n the line of sight and
  u = v × n. Images are `vtkImageData` with a direction matrix (columns
  u, v, n). Pixel area is `(size/(res-1))²`.
- Header particle counts can be larger than what's stored. Cutout snapshots
  keep `NumPart_Total` but drop groups. Trust the groups, not the header.
- `pvgizmo/__init__.py` must be imported before numba or meshoid in anything
  that runs inside ParaView (see the multiprocessing gotcha below).

## ParaView gotchas (all hit in practice)
- `sys.executable` is pvpython/paraview, and numba's parallel init spawns a
  multiprocessing helper. `pvgizmo` calls
  `multiprocessing.set_executable($PV_GIZMO_PYTHON)`.
- The reader's output object is **reused** between executions. Call
  `Initialize()` at the start of `RequestData`, or blocks accumulate.
- `ExtractBlock`'s default `Assembly` is `Hierarchy` if the input hasn't run
  yet, and the reader's own `Assembly` if it has. That's why the root is named
  `Root`.
- The plugin is auto-loaded via `PV_PLUGIN_PATH`, so `LoadPlugin(..., ns=d)`
  then adds nothing to `d`. Use `paraview.simple.GIZMOSnapshotReader`.
  `paraview.simple` builds function names from the XML *label* with spaces
  removed.
- A syntax error in a Python plugin makes it silently not load: the function
  just goes missing. Run `python -c "import ast; ast.parse(open(f).read())"`
  when in doubt.
- GUI scripting (`paraview --script=... --exit`):
  - ParaView 6's modal **Welcome dialog** blocks the script. `test_gui.py`
    passes an isolated `XDG_CONFIG_HOME` whose `ParaView.ini` contains
    `[General] GeneralSettings.ShowWelcomeDialog=false`.
  - `sys.stderr` goes to the GUI's output window, so tracebacks are invisible.
    Write them to `sys.__stdout__` and `os._exit(1)`.
  - `print` needs `flush=True`, or `--exit` drops the output.
  - Data information is empty until something calls `UpdatePipeline`.
  - Run the GUI with `start_new_session` and kill the process group on
    timeout. `timeout xvfb-run ...` leaves orphaned `paraview`/`Xvfb`
    processes behind. Clean up with `pkill -x .paraview-wrapp; pkill -x Xvfb`.
    Don't use `pkill -f` with a pattern that also matches your own shell.
- No `<` or `>` in plugin XML strings, including Documentation. The proxy
  silently fails to register (`test_all_proxies_registered`).
- Volume rendering ignores log-scaled colour maps. Render `log10_*` arrays
  instead (`pvhelpers.show_volume`).
- A filter's array-selection default only applies if its input has already
  run. Call `reader.UpdatePipeline()` before creating filters in scripts.
- `paraview.util.vtkAlgorithm` decorators take the number of elements from
  the method signature. Wrappers with `*args` need an explicit
  `number_of_elements`.
- **Interactivity (`pvgizmo.live`)** runs on the *client*. It observes the
  camera's ModifiedEvent (ignoring clipping-range-only changes, never
  re-entering), and Start/EndInteractionEvent on **the view proxy** (the Qt
  GUI reports them there, not on `view.GetInteractor()`). It sets filter
  properties before the frame renders. GUI detection is `live.in_gui()`
  (`/proc/self/exe`; override with `PV_GIZMO_AUTO_FOLLOW`). The auto-linker
  is installed from the filter constructor, because at plugin import there
  is no session yet. pvpython never auto-follows, so scripted windows are
  never overridden.
- `Delete(proxy)` leaves a usable-looking Python object behind. Check
  liveness with `ProxyManager().SMProxyManager.IsProxyInGroup(p.SMProxy, "sources")`.
- `Xvfb` itself is not on PATH, only `xvfb-run`. Run multi-process GUI tests
  inside one `xvfb-run` (see `tests/gui_drag.sh`).
- Point Gaussian properties in 6.0: `SetScaleArray`, `ScaleByArray`,
  `UseScaleFunction`, `GaussianRadius`, `ShaderPreset`. There is no
  `ScaleArray`.
- Any GUI-script call that spins the Qt event loop (for example
  `scene.UpdateAnimationUsingDataTimeSteps()`, or setting `AnimationTime`) is
  where the Welcome dialog appears. If a GUI test hangs, attach gdb before
  blaming ParaView:
  `gdb -p $(pgrep -x .paraview-wrapp) -batch -ex 'thread apply all bt 25'`.

## Git
- Small, focused commits, and keep the tests green. End commit messages with
  the `Co-Authored-By` trailer used in the history.
- Never commit snapshots, screenshots, `meshoid/` or `testdata/` (see
  `.gitignore`).
