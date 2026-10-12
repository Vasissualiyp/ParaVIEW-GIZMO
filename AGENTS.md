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
plugin/              ParaView Python plugins (auto-loaded via PV_PLUGIN_PATH in the shell)
  GIZMOReader.py     "GIZMO Snapshot Reader" -> vtkPartitionedDataSetCollection
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
- `test_reader_pv.py` covers the reader through the server manager.
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
