#!/usr/bin/env bash
# Run inside xvfb-run (tests/test_gui.py::test_live_mouse_drag): start the GUI on
# tests/gui_live.py, wait until the scene is ready, then drag the mouse over the
# render view with xdotool. Needs PV_GIZMO_LIVE_LOG, SNAPFILE, XDG_CONFIG_HOME.
HERE="$(cd "$(dirname "$0")" && pwd)"
paraview --script="$HERE/gui_live.py" & PP=$!
trap 'kill $PP 2>/dev/null' EXIT
for _ in $(seq 120); do grep -q '"ready"' "$PV_GIZMO_LIVE_LOG" 2>/dev/null && break; sleep 0.5; done
grep -q '"ready"' "$PV_GIZMO_LIVE_LOG" || { echo "GUI never became ready"; exit 1; }
sleep 2
W=$(xdotool search --onlyvisible --name 'ParaView' | head -1)
eval "$(xdotool getwindowgeometry --shell "$W")"
CX=$((X + WIDTH * 65 / 100)); CY=$((Y + HEIGHT / 2))  # inside the render view
xdotool mousemove $CX $CY mousedown 1
for k in $(seq 1 10); do xdotool mousemove $((CX + 15 * k)) $((CY + 3 * k)); sleep 0.15; done
xdotool mouseup 1
sleep 3
echo "GUI_DRAG_DONE"
