# ParaView macro: centre the selected GIZMO reader on its most massive star in every
# snapshot (Recenter = Most massive star) and aim the camera at it: focal point and
# rotation centre on the star, parallel projection, so zooming closes in on the star.
# Use this instead of Reset Camera (which re-centres on the whole box).
import paraview.simple as pvs
from pvgizmo import pvhelpers

src = pvs.GetActiveSource()
if src is not None and src.GetXMLName() == "GIZMOSnapshotReader":
    pvhelpers.center_on_star(src)
pvhelpers.star_camera(pvs.GetActiveView())
