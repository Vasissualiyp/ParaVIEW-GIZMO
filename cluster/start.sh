#!/usr/bin/env bash
# start.sh: one command for a Trillium ParaView session, run on your workstation.
#
#   cluster/start.sh                       # debug partition, 1 h
#   cluster/start.sh -p compute -t 4:00:00 # any extra sbatch options
#
# 1. Logs in to Trillium if there is no ssh master connection yet (asks for MFA once;
#    ControlPersist keeps it for 6 h).
# 2. Submits cluster/pvserver.sbatch unless a pvserver job is already running.
# 3. Waits until pvserver accepts connections, then tunnels and starts the GUI
#    (cluster/connect.sh).
set -euo pipefail

HOST=${PV_HOST:-trillium}
PV_DIR=${PV_DIR:-/scratch/vasissua/ParaVIEW}
cd "$(dirname "$0")/.."

if ! ssh -O check "$HOST" 2>/dev/null; then
    echo "Logging in to $HOST (MFA)..."
    ssh "$HOST" true   # ControlMaster auto + ControlPersist (~/.ssh/config) keep it open
fi

read -r NODE PORT JOB < <(ssh "$HOST" cat "$PV_DIR/pvserver_node.txt" 2>/dev/null) || true
if [ -n "${JOB:-}" ] && ssh "$HOST" squeue -h -j "$JOB" -o %T 2>/dev/null | grep -q RUNNING; then
    echo "Reusing running pvserver job $JOB on $NODE"
else
    ssh "$HOST" rm -f "$PV_DIR/pvserver_node.txt"
    JOB=$(ssh "$HOST" sbatch --parsable "$@" "$PV_DIR/pvserver.sbatch")
    echo "Submitted job $JOB; waiting for it to start..."
    until ssh "$HOST" test -s "$PV_DIR/pvserver_node.txt"; do
        state=$(ssh "$HOST" "squeue -h -j $JOB -o '%T %r'" 2>/dev/null || true)  # ssh drops quoting
        [ -z "$state" ] && { echo "Job $JOB is gone; see $PV_DIR/logs/pvserver-$JOB.out" >&2; exit 1; }
        echo "  $state"
        sleep 10
    done
fi

LOG="$PV_DIR/logs/pvserver-$JOB.out"
until ssh "$HOST" grep -q "Accepting connection" "$LOG" 2>/dev/null; do sleep 3; done
exec cluster/connect.sh
