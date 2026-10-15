#!/usr/bin/env bash
# connect.sh: tunnel to the running Trillium pvserver and start the GUI on it.
#
#   ssh trillium            # once, to pass MFA (ControlPersist keeps the connection 6h)
#   cluster/connect.sh      # from the repo root: tunnel + paraview --server-url=...
#   cluster/connect.sh --tunnel-only
#
# The server writes "<node> <port> <jobid>" to $PV_DIR/pvserver_node.txt
# (cluster/pvserver.sbatch). The forward goes through the ssh master connection, so
# no further logins are needed.
set -euo pipefail

HOST=${PV_HOST:-trillium}
PV_DIR=${PV_DIR:-/scratch/vasissua/ParaVIEW}
LPORT=${PV_LOCAL_PORT:-11111}

if ! ssh -O check "$HOST" 2>/dev/null; then
    echo "No ssh master connection to $HOST: run 'ssh $HOST' once (MFA), then rerun." >&2
    exit 1
fi

read -r NODE RPORT JOB < <(ssh "$HOST" cat "$PV_DIR/pvserver_node.txt" 2>/dev/null || true)
if [ -z "${NODE:-}" ]; then
    echo "No pvserver running ($PV_DIR/pvserver_node.txt missing): submit cluster/pvserver.sbatch." >&2
    exit 1
fi
echo "pvserver: job $JOB on $NODE:$RPORT"

# Replace a forward left over from an earlier session (it may point at another node).
if [ -f "${XDG_RUNTIME_DIR:-/tmp}/pvgizmo-forward" ]; then
    ssh -O cancel -L "$(cat "${XDG_RUNTIME_DIR:-/tmp}/pvgizmo-forward")" "$HOST" 2>/dev/null || true
fi
FWD="$LPORT:$NODE:$RPORT"
ssh -O forward -L "$FWD" "$HOST"
echo "$FWD" > "${XDG_RUNTIME_DIR:-/tmp}/pvgizmo-forward"
echo "tunnel: localhost:$LPORT -> $NODE:$RPORT"

[ "${1:-}" = "--tunnel-only" ] && exit 0
cd "$(dirname "$0")/.."
exec nix develop -c paraview --server-url="cs://localhost:$LPORT"
