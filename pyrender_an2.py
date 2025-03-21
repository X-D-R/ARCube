import cv2 as cv
import numpy as np
import os
import pyrender
import trimesh
from src.detection.detection import detect_pose

MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


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

        pose_obj = np.eye(4)

        self.mesh_node = pyrender.Node(mesh=self.mesh, matrix=pose_obj)
        self.scene.add_node(self.mesh_node)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

        self.cam_node = pyrender.Node(camera=cam)
        self.scene.add_node(self.cam_node)

        light = pyrender.PointLight(color=np.ones(3), intensity=5.0)
        self.light_node = pyrender.Node(light=light)
        self.scene.add_node(self.light_node)

    def update_pose(self, rvecs, tvec):
        if self.mesh_node is None:
            raise ValueError('Launch setup_scene first')

        pose_obj = np.eye(4)
        pose_obj[:3, :3] = rvecs
        pose_obj[:3, 3] = tvec.flatten()

        self.scene.set_pose(self.light_node, pose_obj)

        transform = np.eye(4)
        transform[1, 1] = -1
        transform[2, 2] = -1
        pose_obj = transform @ pose_obj

        self.scene.set_pose(self.mesh_node, pose_obj)

    def render(self):
        # pyrender.Viewer(self.scene, use_ambient_lighting=True, viewer_flags={"show_world_axis": True, "cull_faces": False})
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
    # rendered = cv.resize(rendered, (frame.shape[1], frame.shape[0]))
    # alpha = 0.4
    # new_frame = cv.addWeighted(frame, 1 - alpha, rendered, alpha, 0)
    # return new_frame
    # Приводим изображение к нужному размеру
    rendered = cv.resize(rendered, (frame.shape[1], frame.shape[0]))

    # Создаем маску: считаем белые пиксели как фон (если белый фон, он близок к 255,255,255)
    lower_white = np.array([200, 200, 200], dtype=np.uint8)  # Нижний порог белого
    upper_white = np.array([255, 255, 255], dtype=np.uint8)  # Верхний порог белого
    mask = cv.inRange(rendered, lower_white, upper_white)  # Создаем бинарную маску (белое=255, остальное=0)

    # Инвертируем маску (теперь объект = 1, фон = 0)
    mask_inv = cv.bitwise_not(mask)

    # Преобразуем в трехканальную маску
    mask_inv_3ch = cv.merge([mask_inv, mask_inv, mask_inv])

    # Оставляем только область объекта
    rendered_fg = cv.bitwise_and(rendered, mask_inv_3ch)

    # Оставляем фон без объекта
    frame_bg = cv.bitwise_and(frame, cv.bitwise_not(mask_inv_3ch))

    # Накладываем изображение без фона
    final = cv.add(frame_bg, rendered_fg)

    return final


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


def render_photo(model_path, frame_path, obj_path, cam_path=None, output_path=None):
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
    renderer.load_obj(obj_path)
    renderer.setup_scene(camera_matrix)

    img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)
    valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)

    result = []
    if valid:
        frame = cv.polylines(frame, [np.int32(img_points)], True, 255, 3, cv.LINE_AA)
        rendered = render_frame(renderer, rvecs, tvec)
        result = add_obj(frame, rendered)
        show_save_image(result, output_path)

    return result


def render_video(model_path, video_path, obj_path, cam_path=None, output_path='render_result.mp4'):
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

        if photo:
            render_photo(model_path1, frame_path_second1, obj_path1, cam_path=cam_path1, output_path=output_path_img1)
        else:
            render_video(model_path1, video_path1, obj_path1, cam_path=cam_path1, output_path=output_path_video1)

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

        if photo:
            render_photo(model_path2, frame_path_second2, obj_path2, cam_path=cam_path2)
        else:
            render_video(model_path2, video_path2, obj_path2, cam_path=cam_path2,
                         output_path=output_path_video2)


if __name__ == "__main__":
    from detect import set_detector

    main(sample=2)
