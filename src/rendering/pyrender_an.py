import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os.path

MAIN_DIR = os.path.dirname(os.path.abspath("register.py"))

print(os.path.join(MAIN_DIR, "ExampleFiles", "3d_models", "bus.obj"))
fuze_trimesh = trimesh.load(os.path.join(MAIN_DIR, "ExampleFiles", "3d_models", "bus.obj"))
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)