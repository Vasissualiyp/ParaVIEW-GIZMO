# ParaView macro: for the selected GIZMO reader, one built-in Point Gaussian item per
# particle type (each with its own eye; dark matter starts hidden). Gas is sized by its
# smoothing length, coloured by log10_Density and opaque where dense: edit the opacity
# curve in the Properties panel (Opacity Transfer Function) and colours in the Color
# Map Editor.
import paraview.simple as pvs
from pvgizmo import pvhelpers

pvhelpers.particle_look(pvs.GetActiveSource(), pvs.GetActiveView())
