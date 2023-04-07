import numpy as np
import pyrender

# Define the vertices of the line segments
vertices = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)

# Define the indices that define the line segments
indices = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
], dtype=np.uint32)

import pdb; pdb.set_trace()
# Create the LineSet object
line_set = pyrender.Mesh.from_lines(vertices, indices)

# Create a scene and add the line set to it
scene = pyrender.Scene()
scene.add(line_set)

# Create a viewer to visualize the scene
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
