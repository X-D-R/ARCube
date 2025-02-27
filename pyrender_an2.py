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
        mesh = trimesh.load_mesh(obj_path)
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.mesh = mesh

    def render(self, pose_obj, camera_matrix):
        if self.mesh is None:
            raise ValueError("3D-model is not loaded!")

        mesh_node = pyrender.Node(mesh=self.mesh, matrix=pose_obj)
        self.scene.add_node(mesh_node)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        pose_cam = np.eye(4)
        pose_cam[:3, 3] = np.array([0, -0.14, 0])
        pose_cam[1, 1] = -1
        pose_cam[2, 2] = -1
        print(pose_cam)
        cam_node = pyrender.Node(camera=cam, matrix=pose_cam)
        self.scene.add_node(cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_node = pyrender.Node(light=light, matrix=pose_cam)
        self.scene.add_node(light_node)

        # pyrender.Viewer(self.scene, use_raymond_lighting=True)

        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

        self.scene.remove_node(cam_node)
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(light_node)

        if color.shape[-1] == 4:
            color = cv.cvtColor(color, cv.COLOR_RGBA2RGB)
        color = cv.cvtColor(color, cv.COLOR_RGB2BGR)

        return color

    def render_with_pose(self, rvecs, tvec, camera_matrix):
        if self.mesh is None:
            raise ValueError("3D-модель не загружена!")

        pose_obj = np.eye(4)
        pose_obj[:3, :3] = rvecs
        pose_obj[:3, 3] = tvec .flatten()

        color = self.render(pose_obj, camera_matrix)

        return color

def rendering():
    print('setting detector')

    cam_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
    model_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_script_test.npz'
    frame = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\new_book_check\\book_3.jpg')
    # frame = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\examples\\images\\new_book_check.png')
    obj_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\colored_box.obj'

    detector = set_detector(model_path, cam_path)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    print('loading obj')
    rend = RenderPyrender(frame.shape[1], frame.shape[0])
    rend.load_obj(obj_path)

    print('detecting pose')
    img_points = 1
    # img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)

    if img_points is not None:

        # valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
        # frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)

        valid = True
        rvecs_ref = np.array([[ 0.99928102,  0.01074149,  0.03636013],
                        [-0.00892128,  0.99871648, -0.0498579],
                        [-0.03684901,  0.04949768,  0.99809425]])
        tvec_ref = np.array([ -0.07684176,   -0.10242596,  0.176132])

        # valid = True
        # rvecs_ref = np.array([[ 0.99701514,  0.07669297, -0.00888824],
        #                 [-0.07621702,  0.99606621,  0.04520042],
        #                 [0.01231983, -0.04438806,  0.9989384]])
        # tvec_ref = np.array([ -0.24818897,   -0.30331281,  0.58025545])


        if valid:
            print('rendering')
            rendered = rend.render_with_pose(rvecs_ref, tvec_ref, camera_matrix)
            rendered = cv.resize(rendered, (frame.shape[1], frame.shape[0]))

            alpha = 0.4
            frame = cv.addWeighted(frame, 1 - alpha, rendered, alpha, 0)

    max_height = 800
    h, w, channels = frame.shape
    if h > max_height:
        scale = max_height / h
        frame = cv.resize(frame, (int(w * scale), int(h * scale)))
    cv.imwrite('rendered.jpg', frame)
    cv.imshow("Press enter to close", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

def solving():
    cam_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
    model_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_script_test.npz'
    frame = cv.imread('E:\\pycharm projects\\ARC\\ExampleFiles\\examples\\images\\new_book_check.png')

    detector = set_detector(model_path, cam_path)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)

    pose = np.zeros((3, 4))
    pose[:3, :3] = rvecs
    pose[:3, 3] = tvec.flatten()

    points = [[0, 0.03, 0],
    [0.14, 0.03, 0],
    [0.14, 0.03, -0.21],
    [0, 0.03, -0.21],
    [0, 0, 0],
    [0.14, 0, 0],
    [0.14, 0, -0.21],
    [0, 0, -0.21]]
    for point in points:
        point.append(1)
        print('3d', point)
        print('2d',camera_matrix @ pose @ np.array(point))


rendering()