#!/bin/bash
# status.sh: queue state of the pvserver jobs -> logs/status.txt (run on the login node).
PV_DIR=${PV_DIR:-/scratch/vasissua/ParaVIEW}
{
    date
    squeue --start -u "$USER" -n pvgizmo -o "%.10i %.9P %.2t %.20S %.10M %.10L %R" 2>&1
    echo "node file: $(cat "$PV_DIR/pvserver_node.txt" 2>/dev/null || echo none)"
} > "$PV_DIR/logs/status.txt"
