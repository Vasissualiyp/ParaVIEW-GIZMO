#!/bin/bash
# pvserver.sbatch: a GIZMO-plugin pvserver on one Trillium compute node.
#
# Submit (from the login node, e.g. through the queue_runner):
#   sbatch /scratch/vasissua/ParaVIEW/pvserver.sbatch
# For longer sessions, override on the command line: sbatch -p compute -t 4:00:00 ...
# The debug partition allows one job per user: a second sbatch fails until the first ends.
#
# Writes "<node> <port> <jobid>" to $PV_DIR/pvserver_node.txt once the server is up,
# so the client can tunnel to it (cluster/connect.sh). pvserver exits whenever a
# client disconnects; the loop restarts it, so you can reconnect until the job ends.
#SBATCH --job-name=pvgizmo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --partition=debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --output=/scratch/vasissua/ParaVIEW/logs/pvserver-%j.out

PV_DIR=${PV_DIR:-/scratch/vasissua/ParaVIEW}
PORT=${PV_PORT:-11111}
cd "$PV_DIR" || exit 1
mkdir -p numba-cache logs
echo "== $(date) job $SLURM_JOB_ID on $(hostname), port $PORT"

trap 'rm -f "$PV_DIR/pvserver_node.txt"' EXIT
echo "$(hostname -s) $PORT $SLURM_JOB_ID" > pvserver_node.txt
while true; do
    apptainer run --containall \
        --bind /scratch:/scratch \
        --bind "$PV_DIR/numba-cache:/tmp/numba-cache" \
        "$PV_DIR/pvgizmo.sif" --server-port="$PORT" --force-offscreen-rendering
    echo "== $(date) pvserver exited ($?), restarting"
    sleep 2
done
