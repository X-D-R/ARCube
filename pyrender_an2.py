import pyrender
import trimesh
import numpy as np
import cv2 as cv
from src.detection.detection import detect_pose
from detect import set_detector


class RenderPyrender:
    def __init__(self, w=640, h=480):
        self.scene = pyrender.Scene()
        self.renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        self.mesh = None

    def load_obj(self, obj_path):
        """ Load 3D-model """
        mesh = trimesh.load_mesh(obj_path)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.mesh = mesh

    def render(self, rvecs, tvecs, camera_matrix):
        """ Render 3D-object using pose """
        if self.mesh is None:
            raise ValueError("3D-model is not loaded!")

        pose_obj = np.eye(4)
        pose_obj[:3, :3] = rvecs
        pose_obj[:3, 3] = tvecs.flatten()
        # correction_rotation = np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, -1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ])
        # pose_obj = correction_rotation @ pose_obj
        print(pose_obj)

        mesh_node = pyrender.Node(mesh=self.mesh, matrix=pose_obj)
        self.scene.add_node(mesh_node)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        pose_cam = np.eye(4)
        pose_cam[:3, 3] = np.array([0, 0, -0.21])
        pose_cam[1, 1] = -1
        pose_cam[2, 2] = -1
        print(pose_cam)

        cam_node = pyrender.Node(camera=cam, matrix=pose_cam)
        self.scene.add_node(cam_node)

        # pyrender.Viewer(self.scene, use_raymond_lighting=True)

        color, _ = self.renderer.render(self.scene)
        self.scene.remove_node(cam_node)

        return color

print('setting detector')

cam_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
model_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_script_test.npz'
frame = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\new_book_check\\book_3.jpg')
frame = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\examples\\images\\new_book_check.png')
obj_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\box.obj'

detector = set_detector(model_path, cam_path)
camera_matrix = detector.camera_params['mtx']
dist_coeffs = detector.camera_params['dist']

print('loading obj')
rend = RenderPyrender(frame.shape[1], frame.shape[0])
rend.load_obj(obj_path)

print('detecting pose')
# img_points = 1
img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)

if img_points is not None:

    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
    frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)

    # valid = True
    # rvecs = np.array([[ 0.99928102,  0.01074149,  0.03636013],
    #                 [-0.00892128,  0.99871648, -0.0498579],
    #                 [-0.03684901,  0.04949768,  0.99809425]])
    # tvec = np.array([ -0.07684176,   -0.10242596,  0.176132])

    if valid:
        print('rendering')
        rendered = rend.render(rvecs, tvec, camera_matrix)
        rendered = cv.resize(rendered, (frame.shape[1], frame.shape[0]))
        rendered = cv.cvtColor(rendered, cv.COLOR_RGB2BGR)
        alpha = 0.4
        frame = cv.addWeighted(frame, 1 - alpha, rendered, alpha, 0)

max_height = 800
h, w, channels = frame.shape
if h > max_height:
    scale = max_height / h
    result = cv.resize(frame, (int(w * scale), int(h * scale)))

cv.imshow("Press enter to close", result)
cv.waitKey(0)
cv.destroyAllWindows()
