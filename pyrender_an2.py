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
        self.mesh_node = None
        self.cam_node = None
        self.light_node = None

    def load_obj(self, obj_path):
        mesh = trimesh.load_mesh(obj_path)
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    def setup_scene(self, camera_matrix):
        if self.mesh is None:
            raise ValueError('3D-model is not loaded!')

        self.mesh_node = pyrender.Node(mesh=self.mesh, matrix=np.eye(4))
        self.scene.add_node(self.mesh_node)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

        pose_cam = np.eye(4)
        pose_cam[:3, 3] = np.array([0.07, 0.015, 0.105])
        pose_cam[:3, :3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        self.cam_node = pyrender.Node(camera=cam, matrix=pose_cam)
        self.scene.add_node(self.cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.light_node = pyrender.Node(light=light, matrix=pose_cam)
        self.scene.add_node(self.light_node)

    def update_pose(self, rvecs, tvec):
        if self.mesh_node is None:
            raise ValueError('Launch setup_scene first')

        transform = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        pose_obj = np.eye(4)
        pose_obj[:3, :3] = rvecs @ transform
        pose_obj[:3, 3] = tvec.flatten()
        pose_obj[:3, 3] += np.array([0.07, 0.21, 0.105])

        self.scene.set_pose(self.mesh_node, pose_obj)

    def render(self):
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

    if output_path is not None:
        cv.imwrite(output_path, image)

    if photo:
        cv.imshow("Press enter to close", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        cv.imshow("Detected video. Press 'q' to close", image)
        if cv.waitKey(33) & 0xFF == ord('q'):
            return


def render_photo(cam_path, model_path, frame_path, obj_path, output_path=None):
    frame = cv.imread(frame_path)
    detector = set_detector(model_path, cam_path)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    renderer = RenderPyrender(frame.shape[1], frame.shape[0])
    renderer.load_obj(obj_path)
    renderer.setup_scene(camera_matrix)

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
    print(rvecs)
    print(tvec)

    result = []
    if valid:
        frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)
        rendered = render_frame(renderer, rvecs, tvec)
        result = add_obj(frame, rendered)
        show_save_image(result, output_path)

    return result


def render_video(cam_path, model_path, video_path, obj_path, output_path):
    detector = set_detector(model_path, cam_path)
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('error: video can not be opened')

    renderer = RenderPyrender
    video = None
    first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        if first_frame:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv.CAP_PROP_FPS))
            height, width, channels = frame.shape

            video = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            renderer = RenderPyrender(width, height)
            renderer.load_obj(obj_path)
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
        print("Saved video")
    cv.destroyAllWindows()


def main(photo=True):
    cam_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\CameraParams\\CameraParams.npz'
    model_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\ModelParams\\model_script_test.npz'
    # frame_path_ref = 'E:\\pycharm projects\\ARC\\ExampleFiles\\new_book_check\\book_3.jpg'
    frame_path_second = 'E:\\pycharm projects\\ARC\\ExampleFiles\\examples\\images\\new_book_check.png'
    video_path = 'ExampleFiles\\new_book_check\\new_book_video_main.mp4'
    obj_path = 'E:\\pycharm projects\\ARC\\ExampleFiles\\3d_models\\colored_box.obj'
    output_path_img = 'pyrender_result.jpg'
    output_path_video = 'pyrender_result.mp4'

    if photo:
        render_photo(cam_path, model_path, frame_path_second, obj_path, output_path_img)
    else:
        render_video(cam_path, model_path, video_path, obj_path, output_path_video)


main()
