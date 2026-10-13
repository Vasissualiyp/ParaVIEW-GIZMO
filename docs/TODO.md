# Deferred work

## Phase 4 leftovers (deferred 2026-10-12; the prototype comes first)

Phase 4 is done for a single workstation. These parts are open:

1. **Real-data profiling at scale.** The timings in FINDINGS (Phase 4) come
   from synthetic snapshots (`bench/bench.py`, 1M–16M particles). The real
   FIRE m12f cutouts were only used to check correctness, because they're read
   over sshfs at ~10 MB/s. STARFORGE data hasn't been tried at all. Best done
   on Trillium, where the data is on local disk: run `bench/bench.py`-style
   timings on a full-size FIRE snapshot and on a STARFORGE snapshot.
2. **Trillium verification.** `docs/TRILLIUM.md` has only been verified
   locally (Apptainer with `--containall`). Still to check: module names,
   partition/account flags, compute-node hostnames, the SSH tunnel to a
   compute node, and whether `$SCRATCH` binds work. Needs the user: it
   requires cluster logins.
3. **GPU (CUDA) backend.** Blocked by "Driver/library version mismatch"
   (`nvidia-smi`); reboot first. `numba-cuda` is in the pinned nixpkgs. The
   main payoff would be full-resolution (512²+) frames at ≥16M particles.
   Port `_project2d_core` first (one thread per particle with atomic adds,
   or tile-binned).
4. **Distributed MPI processing.** Today, under `mpiexec pvserver`, only
   rank 0 reads and computes (correct, no speedup). Plan: the reader splits
   by file parts / particle ranges per piece, each rank deposits its own
   particles, and the maps are summed with an allreduce, e.g. by delivering
   per-rank images and adding them in a reduction filter. The meshoid
   slice/derived fields need neighbour (ghost) particles across ranks, which
   is much harder. Only worth it when one node isn't enough.
5. **C++ reader / deposition.** Not needed: profiling found no bottleneck
   that numba doesn't cover. Revisit only if (1) shows the HDF5 read or the
   Python glue dominating.

## Phase 6 leftovers

- **Real FIRE validation of the starlight image.** The only z=0 FIRE snapshot
  in COPY is 7.6 GB (2025-07 `snapdir_590`). Compare against
  `pfh_python.visualization.image_maker` on the same snapshot, on Trillium.
- **Extinction of the glowing point stars** (vizmo `_update_star_columns`).
  The ray-traced Starlight Image already includes extinction; the Point
  Gaussian stars don't.
- **pfh "lighting" layer and three-band gas images** (temperature phases as
  colours: `gas_raytrace_temperature` plus `overlay_lighting_layer`). The ray
  trace already supports any three weights, so this is mostly a filter mode.
- **Baked PSF profiles** (vizmo's UBVRI PSF texture) instead of the analytic
  core/halo/spike shader.
- **Gas-only Metallicity.** The reader loads the full 11–14 component array
  for dust. A "total Z only" option would save memory on huge snapshots.
- **Light levels across time series.** The automatic brightness is per frame;
  movies want fixed levels (set Saturation/DynamicRange, or add explicit
  levels).

## Phase 5 (packaging), deferred: prototype first

A Nix package (`nix run .#paraview-gizmo`) with the plugins preloaded, a user
guide, and optionally a VisIt route. See SPEC.md, Phases.
