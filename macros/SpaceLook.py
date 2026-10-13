# ParaView macro (Macros > Import new macro...): space look for the selected GIZMO reader.
# Black view; FIRE-style starlight image when the snapshot has star particles, else glowing
# gas (column density x temperature); glowing stars for STARFORGE sinks (Type5).
# The images follow the camera.
import paraview.simple as pvs
from pvgizmo import pvhelpers

pvhelpers.space_look(pvs.GetActiveSource())
