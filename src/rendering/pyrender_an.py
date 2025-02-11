import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt


fuze_scene = trimesh.load('E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\bus.obj')
fuze_trimesh = fuze_scene.dump(concatenate=True)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)