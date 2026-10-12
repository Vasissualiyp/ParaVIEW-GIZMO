# ParaView macro (Macros > Import new macro...): toggle live camera-following for the
# selected Meshoid Projection / Slice in the active render view. (Projections created
# in the GUI follow the camera already when their FollowCamera property is on.)
import paraview.simple as pvs
from pvgizmo import live

f = pvs.GetActiveSource()
if live.following(f):
    live.unfollow(f)
    f.FollowCamera = 0
else:
    f.FollowCamera = 1
    live.follow_camera(f)
pvs.Render()
