# ParaView macro: volume render of the selected GIZMO reader's gas, dense gas opaque and
# diffuse gas transparent (Meshoid Grid of the mass density, following the camera).
# Tune the look in the Color Map Editor (opacity curve over 0-1 = DynamicRange decades).
import paraview.simple as pvs
from pvgizmo import pvhelpers

pvhelpers.add_dense_volume(pvs.GetActiveSource(), pvs.GetActiveView())
