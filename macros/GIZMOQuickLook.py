# ParaView macro (Macros > Import new macro...): for the selected GIZMO reader,
# show gas as Point Gaussian splats and add a column-density projection.
import paraview.simple as pvs
from pvgizmo import pvhelpers

src = pvs.GetActiveSource()
src.UpdatePipeline()
pvhelpers.show_particles(src)
pvs.Hide(src)  # the raw reader output; the extracted gas is shown instead
pvhelpers.add_projection(src)
pvs.Render()
