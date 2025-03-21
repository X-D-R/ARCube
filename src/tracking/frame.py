import cv2 as cv
import numpy as np
import os.path
from pyrender_an2 import RenderPyrender, render_frame, add_obj
from rendering_CV import OBJ, render_CV
from src.tracking.klt_tracker_class import KLT_Tracker
from src.detection.detection import detect_pose, Detector
from src.utils.draw_functions import visualize_matches

MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


class FrameRegistration:
    def __init__(self):
        return

    def track_features_sift(self, previous_frame: np.ndarray = None, current_frame: np.ndarray = None,
                            kpoints_2D: np.ndarray = None, kpoints_3D: np.ndarray = None) -> (
            np.ndarray, np.ndarray, np.ndarray):  # track features KLT
        '''
        Track key points using KLT tracker and match 2D and 3D key points

        :param previous_frame: previous frame from video
        :param current_frame: current frame
        :param kpoints_2D: 2D coordinates of key_points from previous frame
        :param kpoints_3D: 3D coordinates of key_points from previous frame
        :return: 2D coordinates of key points from previous and current frames, 3D coordinates of key points from new_frames
        '''

        tracker = KLT_Tracker()
        good_new_kpoints, good_old_kpoints, status = tracker.track(previous_frame, current_frame, kpoints_2D)
        if len(good_new_kpoints) != len(kpoints_3D):  # matching 3D features with 2D features
            new_kpoints_3D = []
            for i in range(len(status)):
                if status[i] == 1:
                    new_kpoints_3D.append(kpoints_3D[i])
            new_kpoints_3D = np.array(new_kpoints_3D)
        else:
            new_kpoints_3D = kpoints_3D

        return good_new_kpoints, good_old_kpoints, new_kpoints_3D

    def find_new_corners(self, kpoints_3D: np.ndarray, kpoints_2D: np.ndarray,
                         cameraMatrix: np.ndarray, distCoeffs: np.ndarray,
                         corners_3D: np.ndarray, detector: Detector = None):
        '''
        Find pose of object and calculate coordinates of corners
        :param kpoints_3D: 3D coordinates of key points
        :param kpoints_2D: 2D coordinates of key points
        :param cameraMatrix: camera matrix
        :param distCoeffs: distortion coefficient
        :param corners_3D: 3D coordinates of corners from model
        :return: 2D coordinates of corners
        '''
        if len(kpoints_2D) > 3 and len(kpoints_2D) == len(kpoints_3D):
            rvec, tvec = detector.get_rvec_tvec() if detector is not None else None, None
            valid, rvecs, tvec = detect_pose(kpoints_2D, kpoints_3D, cameraMatrix, distCoeffs, rvec, tvec)
            corners_3D = np.float32(corners_3D)  # 3D coordinates of corners of model
            corners_2D, _ = cv.projectPoints(corners_3D, rvecs, tvec, cameraMatrix,
                                             distCoeffs)  # transform to 2D coordinates using pose
            corners_2D = corners_2D.reshape(-1, 1, 2)
            # frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)

            return corners_2D, valid, rvecs, tvec
        else:
            return None


def track_frame(detector: Detector, video_path: str = None, output_path: str = None,
                fps: int = 30, color: tuple = (255, 0, 0), visualizing_matches: bool = False,
                use_tracker: bool = False, use_web_camera: bool = False, save_video: bool = False,
                render_cv: bool = False, pyrender: str = None) -> None:
    '''
    This func tracks object on video (with detection every track_length) and save video to output
    :param detector: Detector, detector, that used to detect object
    :param video_path: str, path to original video
    :param output_path: str, destination where video should be saved
    :param fps: int, frame per second (optional), 30 fps is usually used
    :param color: tuple, color of frame in BGR color scheme (255, 0 , 0) - Blue
    :param visualizing_matches: bool, whether to visualize matches with reference image
    :param use_tracker: bool, use tracker or not
    :param use_web_camera: bool, use web_camera or video
    :param save_video: bool, save video or not
    :param render_cv: bool, render 3d model by cv over detected object or not
    :param pyrender: str, path to the obj file if rendering 3d model by pyrender over detected object
    :return: None
    '''
    reference_image = detector.registration_params['img']
    reference_kp = detector.registration_params["key_points_2d"]
    cameraMatrix, distCoeffs = detector.camera_params['mtx'], detector.camera_params['dist']

    if use_web_camera:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(video_path)  # load video file

    tracker = FrameRegistration()

    object_corners_3d = detector.registration_params['object_corners_3d']
    object_corners_2d = detector.registration_params['object_corners_2d']

    ret, previous_frame = cap.read()

    if not ret:
        print("Failed to read the first frame.")
        exit()

    obj, texture = None, None
    if render_cv:
        obj = OBJ(os.path.join(os.path.join(MAIN_DIR, "ExampleFiles", "3d_models", "box_CV.npz")), swapyz=True)
        texture = cv.imread('hse.jpg')
        rend = render_CV()

    renderer = None
    if pyrender is not None:
        height, width, channels = previous_frame.shape
        renderer = RenderPyrender(width, height)
        renderer.load_obj(pyrender)
        renderer.setup_scene(cameraMatrix)

    object_corners_2d, kpoints_3d, kpoints_2d, kp, matches, M, mask = detector.detect(previous_frame)
    mask = np.zeros_like(previous_frame)
    Images = []
    Images_matching = []
    count = 1
    img = previous_frame
    imgSize = None
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if not use_tracker:
            # only detect new frame
            img_pts_detected, kpoints_3d_detected, kpoints_2d_detected, kp_1, matches_1, M_1, mask_1 = detector.detect(
                previous_frame)
            if img_pts_detected is None:
                print("Bad detection of object, keeping old parameters")
            else:
                object_corners_2d, kpoints_3d, kpoints_2d = img_pts_detected, kpoints_3d_detected, kpoints_2d_detected
                kp, matches = kp_1, matches_1
                mask = np.zeros_like(previous_frame)
        else:
            img_pts_detected, kpoints_3d_detected, kpoints_2d_detected, kp_1, matches_1, M_1, mask_1 = detector.detect(
                previous_frame)
            if img_pts_detected is None or kpoints_2d is None:
                print("Bad detection of object, tracking")
                good_new, good_old, kpoints_3d = tracker.track_features_sift(previous_frame, frame, kpoints_2d,
                                                                             kpoints_3d)
                object_corners_2d, valid, rvec, tvec = tracker.find_new_corners(kpoints_3d, good_new, cameraMatrix,
                                                                                distCoeffs,
                                                                                object_corners_3d, detector)
                kpoints_2d = good_new.reshape(-1, 1, 2)
            else:
                object_corners_2d, kpoints_3d, kpoints_2d = img_pts_detected, kpoints_3d_detected, kpoints_2d_detected
                kp, matches = kp_1, matches_1
                mask = np.zeros_like(previous_frame)

        if visualizing_matches:
            Images_matching.append(visualize_matches(reference_image, reference_kp, frame, kp, matches))

        if object_corners_2d is not None:
            frame = cv.polylines(frame, [np.int32(object_corners_2d)], True, color, 3, cv.LINE_AA)
            if render_cv:
                valid, rvec, tvec = detect_pose(kpoints_2d, kpoints_3d, cameraMatrix, distCoeffs)
                if valid:
                    img = rend.render(frame, obj, rvec, tvec, cameraMatrix, distCoeffs, texture)
                if img is None:
                    img = cv.add(frame, mask)
            elif pyrender is not None:
                valid, rvec, tvec = detect_pose(kpoints_2d, kpoints_3d, cameraMatrix, distCoeffs)
                if valid:
                    rendered = render_frame(renderer, rvec, tvec)
                    img = add_obj(frame, rendered)
            else:
                img = cv.add(frame, mask)
        else:
            count = -1
            img = frame
        imgSize = img.shape
        Images.append(img)
        if len(frame) > 0:
            previous_frame = frame.copy()

        count += 1

        if use_web_camera:
            cv.imshow('Press "Esc" to close the programm', img)
            if cv.waitKey(33) == 27:
                cv.destroyAllWindows()
                break

    if visualizing_matches:
        for i in range(len(Images_matching)):
            img = Images_matching[i]
            cv.imshow("Visualizing of matches. Press 'Esc' to close", img)
            if cv.waitKey(33) == 27:
                break
    if save_video:
        # saving video
        height, width, channels = imgSize

        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        video = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(len(Images)):
            video.write(Images[i])
            if (cv.waitKey(1) & 0xFF) == ord('q'):
                break
        video.release()
        print("Saved video")

    max_height = 800
    for i in range(len(Images)):
        img = Images[i]
        h, w, channels = img.shape
        if h > max_height:
            scale = max_height / h
            img = cv.resize(img, (int(w * scale), int(h * scale)))
        cv.imshow("Detected video. Press 'Esc' to close", img)
        if cv.waitKey(33) == 27:
            break

    cv.destroyAllWindows()
