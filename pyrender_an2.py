import numpy as np
import trimesh
import pyrender
import cv2 as cv
import matplotlib.pyplot as plt
from src.tracking.frame import detect_pose
from src.registration.rectangle_model import RectangleModel

fuze_trimesh = trimesh.load('E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\box.obj')

scene = pyrender.Scene()

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 0.3],
    metallicFactor=0.0,
    roughnessFactor=0.5
)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, material=material)
scene.add(mesh)

model = RectangleModel.load('E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_test.npz')
kpoints_2D = model.object_corners_2d
kpoints_3D = model.object_corners_3d

path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
if path.endswith('.npz'):
    with np.load(path) as file:
        cameraMatrix = file['cameraMatrix']
        distCoeffs = file['dist']

valid, rvecs, tvec = detect_pose(kpoints_2D, kpoints_3D, cameraMatrix, distCoeffs)

camera_pose = np.eye(4)
camera_pose[:3, :3] = rvecs
camera_pose[:3, 3] = tvec.ravel()

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
scene.add(camera, pose=camera_pose)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

image = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\examples\\images\\new_book_check.png')
h, w, channels = image.shape

r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
alpha_mask = color[:, :, 3] / 255.0
alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
color = color[:, :, :3]

result = image.copy()
result[alpha_mask > 0] = (color * alpha_mask + result * (1 - alpha_mask))[alpha_mask > 0]


max_height = 800
h, w, channels = result.shape
if h > max_height:
    scale = max_height / h
    result = cv.resize(result, (int(w * scale), int(h * scale)))

cv.imshow("Press enter to close", result)
cv.waitKey(0)
cv.destroyAllWindows()

