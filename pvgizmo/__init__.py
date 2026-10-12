"""ParaView-independent core for the GIZMO/meshoid ParaView plugins."""

import multiprocessing
import os

# sys.executable is pvpython/paraview inside ParaView, which cannot run
# multiprocessing helpers; numba's parallel init spawns one (see docs/FINDINGS.md).
# Must run before numba/meshoid are imported.
if os.environ.get("PV_GIZMO_PYTHON"):
    multiprocessing.set_executable(os.environ["PV_GIZMO_PYTHON"])

__version__ = "0.1.0"
