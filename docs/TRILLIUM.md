# Running the GIZMO plugins on a cluster (Trillium) in client–server mode

The data stays on the cluster. A `pvserver` there runs the reader and the
Meshoid filters. Your local ParaView GUI (from this flake) renders, and the
camera-following projections are recomputed on the server as you move the view.

What's verified, and what isn't:
- **Verified locally (2026-10-12):** this image runs under Apptainer with
  `--containall`, so there is no host `/nix`, GPU or X. The GUI and pvpython
  connect to it. Server-side and client-side rendering both work, rotated maps
  keep their orientation, camera-following works, and `mpiexec -n 3 pvserver`
  gives the single-process results (`tests/test_client_server.py`).
- **Verified on Trillium (2026-10-15):** `cluster/pvserver.sbatch` (debug
  partition, rrg-rbond-ac, one whole node) ran the image on `tri0003`.
  Apptainer is in `/usr/bin`, so no module is needed. The tunnel from a workstation
  through the login node reached it, and `cluster/check_remote.py` passed on m12f
  `output_cutout/snapshot_259` (73,825 gas, 23 sinks): 512² projection 1.0 s
  per rotation, space look 2.2 s, a 4× zoom 0.75 s, all including delivery to the
  client. The GUI connected the same way.

## Quick start (the scripted route)

On the workstation, from the repo root:

```sh
cluster/start.sh                          # debug partition, 1 h
cluster/start.sh -p compute -t 4:00:00    # longer: any sbatch options
```

It logs in if needed (MFA once; the ssh ControlPersist keeps the connection
for 6 h), reuses a running pvserver job or submits `pvserver.sbatch`, waits
until the server accepts connections, tunnels, and starts the GUI. If you close
the GUI, rerun `cluster/start.sh` (or `cluster/connect.sh`) to reconnect to the
same job. The debug partition allows one job per user.

`nix develop -c pvpython cluster/check_remote.py` (after
`cluster/connect.sh --tunnel-only`) is a headless end-to-end check.
The job restarts pvserver after each disconnect, so you can reconnect until the
time limit. Rebuild and re-upload the image (section 1) after changing
`pvgizmo/` or `plugin/`.
Sections 1–3 below describe the manual steps.

## 1. Build the server image (on your machine)

The client and server must be the **same ParaView build**: 6.0.1 from this
flake's pinned nixpkgs.

```sh
nix build .#pvserver-image -o pvserver-image          # ~1.6 GB docker-archive (.tar.gz)
gunzip -c pvserver-image > pvgizmo-image.tar          # apptainer wants an uncompressed archive
nix develop -c apptainer build pvgizmo.sif docker-archive://pvgizmo-image.tar   # ~1.5 GB
rsync -P pvgizmo.sif USER@trillium.scinet.utoronto.ca:'$SCRATCH'/
```

The image contains `pvserver` and `pvpython` 6.0.1, the Python env (numpy,
scipy, numba, h5py), meshoid at the flake's pinned commit, `pvgizmo` and the
plugins, and Mesa (EGL with llvmpipe) for offscreen rendering without a GPU.

**Rebuild the image whenever `pvgizmo/` or `plugin/` change.** The plugin
code is baked into it.

## 2. Start pvserver on a compute node

```sh
ssh USER@trillium.scinet.utoronto.ca
salloc --nodes=1 --time=2:00:00            # (partition/account flags as required)
module load apptainer                       # name to be checked: module spider apptainer
hostname                                    # e.g. tri0123 -> needed for the tunnel
apptainer run --bind "$SCRATCH:$SCRATCH" "$SCRATCH/pvgizmo.sif" \
    --server-port=11111 --force-offscreen-rendering
# prints "Accepting connection(s): <node>:11111"
```

- **Run one process per node.** The deposition kernels use all cores through
  numba threads. Under `mpiexec` every rank runs, but only rank 0 reads and
  computes: results stay correct, but there is no speedup yet.
- Set `NUMBA_NUM_THREADS` to limit the threads if you share the node.
- The first projection or slice JIT-compiles (seconds). The numba cache is
  `/tmp/numba-cache` inside the container. To keep it across sessions, bind a
  directory: `--bind $SCRATCH/numba:/tmp/numba-cache`.
- `pvserver` exits when the client disconnects.

## 3. Tunnel and connect (on your machine)

```sh
ssh -N -L 11111:tri0123:11111 USER@trillium.scinet.utoronto.ca     # node from step 2
nix develop -c paraview --server-url=cs://localhost:11111
```

Or use File → Connect → `cs://localhost:11111` from a running GUI. Then:

- **File → Open browses the cluster's file system.** Open e.g.
  `$SCRATCH/.../snapshot_100.hdf5` with the "GIZMO Snapshot Reader". For zoom
  or cutout runs, set **Recenter = Gas density peak** (see FINDINGS: float32
  collapse).
- Filters → GIZMO → **Meshoid Projection**. FollowCamera is on, so rotating,
  zooming or panning re-projects on the server: 128² while you drag, full
  resolution on release.
- **Small results come to your machine for rendering**: 2D maps and grids
  smaller than ParaView's *Remote Render Threshold* (20 MB). Larger ones are
  rendered on the server with Mesa, which is slower. Projections are a few MB
  at most. Avoid showing millions of raw particles: show the maps, or use the
  reader's **Stride** for a particle preview.

## Notes

- Speeds are in docs/FINDINGS.md (Phase 4). One 128² camera move costs
  0.14 s at 16M particles on a 20-core workstation, plus the network
  round-trip for the image.
- If the data is reachable from your machine (e.g. the sshfs mount),
  builtin mode works too. Every read then goes over the network (about
  10 MB/s here), so client–server is the way to go for big snapshots.
- Reverse connections (`pvserver --reverse-connection --client-host=...`) are
  an alternative if compute nodes can reach your machine. They usually can't.
