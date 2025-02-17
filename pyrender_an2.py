import numpy as np
import trimesh
import pyrender
import cv2 as cv
from src.tracking.frame import detect_pose
from src.registration.rectangle_model import RectangleModel
from src.detection.detection import Detector

print('loading 3d model')
box_trimesh = trimesh.load('E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\box_2.obj')
scene = pyrender.Scene()
material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 0.3],
    metallicFactor=0.0,
    roughnessFactor=0.5
)
mesh = pyrender.Mesh.from_trimesh(box_trimesh, material=material)
scene.add(mesh)

print('detecting pose')
image = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\new_book_check\\book_3.jpg')
h, w, channels = image.shape

cam_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
if cam_path.endswith('.npz'):
    with np.load(cam_path) as file:
        cameraMatrix = file['cameraMatrix']
        distCoeffs = file['dist']

model_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_test.npz'
detector = Detector()
detector.set_detector(cam_path, model_path)

img_pts, kpoints_3d, kpoints_2d = detector.detect(image)
valid, rvecs, tvec = detect_pose(kpoints_2d, kpoints_3d, cameraMatrix, distCoeffs)


camera_pose = np.eye(4)
camera_pose[:3, :3] = rvecs
camera_pose[:3, 3] = tvec.ravel()
print(camera_pose)

# X
correction_rotation = np.array([
    [1,  0,  0,  0],
    [0,  0, -1,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1]
])
camera_pose = correction_rotation @ camera_pose
# Y
correction_rotation = np.array([
    [0,  0,  1,  0],
    [0,  1,  0,  0],
    [-1, 0,  0,  0],
    [0,  0,  0,  1]
])
camera_pose = correction_rotation @ camera_pose
# # Z
# correction_rotation = np.array([
#     [0, -1,  0,  0],
#     [1,  0,  0,  0],
#     [0,  0,  1,  0],
#     [0,  0,  0,  1]
# ])
camera_pose = correction_rotation @ camera_pose
print(camera_pose)

print('showing model')
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
scene.add(camera, pose=camera_pose)


# light_positions = [
#     [1, 1, 1],   # Сверху справа
#     [-1, 1, 1],  # Сверху слева
#     [1, -1, 1],  # Снизу справа
#     [-1, -1, 1], # Снизу слева
#     [1, 1, -1],  # Сзади справа
#     [-1, 1, -1], # Сзади слева
# ]
#
# for pos in light_positions:
#     light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#     light_pose = np.eye(4)
#     light_pose[:3, 3] = pos
#     scene.add(light, pose=light_pose)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)

print('showing results')
# 1 way
# alpha_mask = color[:, :, 3] / 255.0
# alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
# color = color[:, :, :3]
# result = image.copy()
# result[alpha_mask > 0] = (color * alpha_mask + result * (1 - alpha_mask))[alpha_mask > 0]

# 2 way
color = cv.cvtColor(color, cv.COLOR_RGB2BGR)
alpha = 0.5
result = cv.addWeighted(image, 1 - alpha, color, alpha, 0)


max_height = 800
h, w, channels = result.shape
if h > max_height:
    scale = max_height / h
    result = cv.resize(result, (int(w * scale), int(h * scale)))

cv.imshow("Press enter to close", result)
cv.waitKey(0)
cv.destroyAllWindows()

