import cv2 as cv
import numpy as np
import os
import pyrender
import trimesh
from detect import set_detector
from src.detection.detection import detect_pose

MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


class RenderPyrender:
    def __init__(self, w=640, h=480):
        self.scene = pyrender.Scene()
        self.renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        self.x = 0
        self.y = 0
        self.z = 0
        self.mesh = None
        self.mesh_node = None
        self.cam_node = None
        self.light_node = None

    def load_obj(self, obj_path, x, y, z):
        mesh = trimesh.load_mesh(obj_path)
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.x, self.y, self.z = x, y, z

    def setup_scene(self, camera_matrix):
        if self.mesh is None:
            raise ValueError('3D-model is not loaded!')

        # Подумать нужно ли что то делать с позой самого объекта!!!!!!!!!!
        pose_obj = np.eye(4)
        # pose_obj[:3, 3] = -np.array([self.x, self.y, self.z])

        self.mesh_node = pyrender.Node(mesh=self.mesh, matrix=pose_obj)
        self.scene.add_node(self.mesh_node)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        print("fx, fy:", camera_matrix[0, 0], camera_matrix[1, 1])
        print("cx, cy:", cx, cy)

        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        # cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        world_axes = trimesh.creation.axis(origin_size=0.05)
        world_mesh = pyrender.Mesh.from_trimesh(world_axes, smooth=False)
        world_axes_node = pyrender.Node(mesh=world_mesh, matrix=np.eye(4))
        self.scene.add_node(world_axes_node)

        self.cam_node = pyrender.Node(camera=cam)
        self.scene.add_node(self.cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.light_node = pyrender.Node(light=light)
        self.scene.add_node(self.light_node)

    def update_pose(self, rvecs, tvec):
        if self.mesh_node is None:
            raise ValueError('Launch setup_scene first')

        pose_cam = np.eye(4)
        pose_cam[:3, :3] = rvecs
        pose_cam[:3, 3] = tvec.flatten()


        # transform = np.eye(4)
        # transform[:3, :3] = np.array([
        #     [-1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, 1]
        # ])
        #
        # pose_cam = transform @ pose_cam

        camera_axes = trimesh.creation.axis(origin_size=0.05)
        camera_mesh = pyrender.Mesh.from_trimesh(camera_axes, smooth=False)
        camera_axes_node = pyrender.Node(mesh=camera_mesh, matrix=pose_cam)
        self.scene.add_node(camera_axes_node)

        self.scene.set_pose(self.cam_node, pose_cam)
        self.scene.set_pose(self.light_node, pose_cam)


    def render(self):
        pyrender.Viewer(self.scene, use_raymond_lighting=True)

        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        if color.shape[-1] == 4:
            color = cv.cvtColor(color, cv.COLOR_RGBA2RGB)
        color = cv.cvtColor(color, cv.COLOR_RGB2BGR)

        return color


def render_frame(renderer: RenderPyrender, rvecs, tvec):
    renderer.update_pose(rvecs, tvec)
    rendered = renderer.render()
    return rendered


def add_obj(frame, rendered):
    rendered = cv.resize(rendered, (frame.shape[1], frame.shape[0]))
    alpha = 0.4
    new_frame = cv.addWeighted(frame, 1 - alpha, rendered, alpha, 0)
    return new_frame


def show_save_image(image, output_path=None, photo=True):
    max_height = 800
    h, w, channels = image.shape
    if h > max_height:
        scale = max_height / h
        image = cv.resize(image, (int(w * scale), int(h * scale)))

    if photo:
        if output_path is not None:
            cv.imwrite(output_path, image)

        cv.imshow("Press enter to close", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        cv.imshow("Detected video. Press 'q' to close", image)
        if cv.waitKey(33) & 0xFF == ord('q'):
            return


def render_photo(model_path, frame_path, obj_path, x, y, z, cam_path=None, output_path=None):
    frame = cv.imread(frame_path)
    camera_params_approximate = {}
    if cam_path is None:
        h, w, channels = frame.shape
        f = 0.9 * max(w, h)
        camera_params_approximate = {'mtx': np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32),
                                     'dist': np.array([0, 0, 0, 0, 0], np.float32)}

    detector = set_detector(model_path, cam_path, camera_params_approximate=camera_params_approximate)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    renderer = RenderPyrender(frame.shape[1], frame.shape[0])
    renderer.load_obj(obj_path, x, y, z)
    renderer.setup_scene(camera_matrix)

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)

    # valid = True
    # rvecs_ref = np.array([[ 0.99928102,  0.01074149,  0.03636013],
    #                 [-0.00892128,  0.99871648, -0.0498579],
    #                 [-0.03684901,  0.04949768,  0.99809425]])
    # tvec_ref = np.array([-0.07684176,   -0.10242596,  0.176132])
    # rvecs, tvec = rvecs_ref, tvec_ref

    # valid = True
    # rvecs_second = np.array([[0.99701514,  0.07669297, -0.00888824],
    #                         [-0.07621702,  0.99606621,  0.04520042],
    #                         [0.01231983, -0.04438806,  0.9989384]])
    # tvec_second = np.array([-0.24818897,   -0.30331281,  0.58025545])
    # rvecs, tvec = rvecs_second, tvec_second

    result = []
    if valid:
        frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)
        rendered = render_frame(renderer, rvecs, tvec)
        result = add_obj(frame, rendered)
        show_save_image(result, output_path)

    return result


def render_video(model_path, video_path, obj_path, x, y, z, cam_path=None, output_path='render_result.mp4'):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('error: video can not be opened')

    detector, camera_matrix, dist_coeffs = None, None, None
    renderer = RenderPyrender
    video = None
    first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Unable to capture video')
            break

        if first_frame:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv.CAP_PROP_FPS))
            height, width, channels = frame.shape

            camera_params_approximate = {}
            if cam_path is None:
                h, w, channels = frame.shape
                f = 0.9 * max(w, h)
                camera_params_approximate = {'mtx': np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32),
                                             'dist': np.array([0, 0, 0, 0, 0], np.float32)}

            detector = set_detector(model_path, cam_path, camera_params_approximate=camera_params_approximate)
            camera_matrix = detector.camera_params['mtx']
            dist_coeffs = detector.camera_params['dist']

            video = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            renderer = RenderPyrender(width, height)
            renderer.load_obj(obj_path, x, y, z)
            renderer.setup_scene(camera_matrix)

            first_frame = False

        img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
        valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)

        if valid:
            frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)
            rendered = render_frame(renderer, rvecs, tvec)
            result = add_obj(frame, rendered)

            video.write(result)
            show_save_image(result, photo=False)

    cap.release()
    if video is not None:
        video.release()
        print('Saved video')
    cv.destroyAllWindows()


def main(photo=True, sample=1):
    if sample == 1:
        cam_path1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'CameraParams', 'CameraParams.npz')
        cam_path1 = None
        model_path1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'ModelParams', 'model_script_test.npz')
        frame_path_ref1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'new_book_check', 'book_3.jpg')
        frame_path_second1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'new_book_check.png')
        video_path1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'new_book_check', 'new_book_video_main.mp4')
        obj_path1 = os.path.join(MAIN_DIR, 'ExampleFiles', '3d_models', 'colored_box.obj')
        output_path_img1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'OutputFiles', 'OutputImages', 'pyrender_result.jpg')
        output_path_video1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'OutputFiles', 'OutputVideos',
                                          'pyrender_result.mp4')
        x1, y1, z1 = 0.14, 0.21, 0.03

        if photo:
            # render_photo(cam_path1, model_path1, frame_path_second1, obj_path, x1, y1, z1, output_path_img1)
            render_photo(model_path1, frame_path_second1, obj_path1, x1, y1, z1, cam_path=cam_path1)
        else:
            render_video(model_path1, video_path1, obj_path1, x1, y1, z1, cam_path=cam_path1,
                         output_path=output_path_video1)

    if sample == 2:
        cam_path2 = None
        model_path2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'ModelParams', 'model_varior_book_iphone.npz')
        frame_path_ref2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'OutputFiles', 'OutputImages',
                                       'varior_book_iphone.jpg')
        frame_path_second2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'varior_book_iphone2.jpg')
        frame_path_third2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'varior_book_iphone3.jpg')
        frame_path_fourth2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'varior_book_iphone4.jpg')
        video_path2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'videos', 'varior_book_iphone2.MOV')
        obj_path2 = os.path.join(MAIN_DIR, 'ExampleFiles', '3d_models', 'colored_box_varior.obj')
        output_path_img2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'OutputFiles', 'OutputImages',
                                        'pyrender_result_varior.jpg')
        output_path_video2 = os.path.join(MAIN_DIR, 'ExampleFiles', 'OutputFiles', 'OutputVideos',
                                          'pyrender_result_varior.mp4')
        x2, y2, z2 = 0.135, 0.02, 0.205
        if photo:
            # render_photo(cam_path1, model_path1, frame_path_second1, obj_path, x1, y1, z1, output_path_img1)
            render_photo(model_path2, frame_path_ref2, obj_path2, x2, y2, z2, cam_path=cam_path2,
                         output_path=output_path_img2)
        else:
            render_video(model_path2, video_path2, obj_path2, x2, y2, z2, cam_path=cam_path2,
                         output_path=output_path_video2)


def test():
    cam_path = os.path.join(MAIN_DIR, 'ExampleFiles', 'CameraParams', 'CameraParams.npz')
    model_path = os.path.join(MAIN_DIR, 'ExampleFiles', 'ModelParams', 'model_script_test.npz')
    frame_path_ref1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'new_book_check', 'book_3.jpg')
    frame_path_second1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'new_book_check.png')
    axes_path = os.path.join(MAIN_DIR, 'ExampleFiles', '3d_models', 'axes.obj')
    new_axes_path = os.path.join(MAIN_DIR, 'ExampleFiles', '3d_models', "axes_boxes.obj")

    frame = cv.imread(frame_path_second1)
    detector = set_detector(model_path, cam_path)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
    if valid:
        frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)
        transform = np.eye(4)
        transform[1, 1] = -1
        transform[2, 2] = -1
        # подумать над трансформ!!!!!!!!!!!

        pose_cam = np.eye(4)
        pose_cam[:3, :3] = rvecs
        pose_cam[:3, 3] = tvec.flatten()
        # pose_cam = transform @ pose_cam


    mesh = trimesh.load_mesh(new_axes_path)

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera = pyrender.PerspectiveCamera(yfov=0.7)
    scene.add(camera, pose=pose_cam)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0)
    scene.add(light, pose=pose_cam)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)


def test_axes():

    cam_path = os.path.join(MAIN_DIR, 'ExampleFiles', 'CameraParams', 'CameraParams.npz')
    cam_path = None
    model_path = os.path.join(MAIN_DIR, 'ExampleFiles', 'ModelParams', 'model_script_test.npz')
    frame_path_ref1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'new_book_check', 'book_3.jpg')
    frame_path_second1 = os.path.join(MAIN_DIR, 'ExampleFiles', 'examples', 'images', 'new_book_check.png')
    obj_path1 = os.path.join(MAIN_DIR, 'ExampleFiles', '3d_models', 'colored_box.obj')

    frame = cv.imread(frame_path_second1)

    h, w, channels = frame.shape
    f = 0.9 * max(w, h)
    camera_params_approximate = {'mtx': np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32),
                                 'dist': np.array([0, 0, 0, 0, 0], np.float32)}

    detector = set_detector(model_path, cam_path, camera_params_approximate=camera_params_approximate)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    mesh = trimesh.load_mesh(obj_path1)

    scene = pyrender.Scene()

    scene.add(pyrender.Mesh.from_trimesh(mesh))

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)

    pose_cam = np.eye(4)
    if valid:
        pose_cam[:3, :3] = rvecs
        pose_cam[:3, 3] = tvec.flatten()

        # transform = np.eye(4)
        # transform[:3, :3] = np.array([
        #     [-1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, 1]
        # ])
        #
        # pose_cam = transform @ pose_cam

    world_axes = trimesh.creation.axis(origin_size=0.05)
    world_mesh = pyrender.Mesh.from_trimesh(world_axes, smooth=False)
    world_axes_node = pyrender.Node(mesh=world_mesh, matrix=np.eye(4))
    scene.add_node(world_axes_node)

    camera_axes = trimesh.creation.axis(origin_size=0.05)
    camera_mesh = pyrender.Mesh.from_trimesh(camera_axes, smooth=False)
    camera_axes_node = pyrender.Node(mesh=camera_mesh, matrix=pose_cam)
    scene.add_node(camera_axes_node)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    cam_node = pyrender.Node(camera=cam)
    scene.add_node(cam_node)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0)
    scene.add(light, pose=pose_cam)

    pyrender.Viewer(scene, use_raymond_lighting=True)

    print("=== Поза камеры (pose_cam) ===")
    print(pose_cam)

    print("=== Позиция объекта относительно камеры ===")
    obj_position = np.linalg.inv(pose_cam) @ np.array([0, 0, 0, 1])
    print("Объект в координатах камеры:", obj_position)

    cam_mtx = np.hstack((camera_matrix, np.zeros((3, 1))))  # Превращаем 3x3 в 3x4

    print("Матрица камеры (K):\n", cam_mtx)

    # Проекция 3D-точки в систему координат камеры
    cam_point = pose_cam @ np.array([0, 0, 0, 1])  # Мировая точка в координатах камеры

    print("Объект в системе координат камеры:", cam_point)

    # Проецируем на 2D-изображение
    img_point = cam_mtx @ cam_point[:4]

    # Нормализуем (делим на z)
    u, v = img_point[:2] / img_point[2]

    print("2D координаты объекта на изображении:", u, v)

main(sample=1)
# test()
# test_axes()
