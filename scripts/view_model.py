"""Open the biped model in MuJoCo's interactive viewer.

Drag to orbit, scroll to zoom, ctrl+drag to apply a force to a body
(useful for poking the robot by hand before we automate it).
"""

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("models/biped.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
