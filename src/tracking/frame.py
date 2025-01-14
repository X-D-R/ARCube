import cv2 as cv
import numpy as np
from src.registration.rectangle_model import register
from src.tracking.klt_tracker_class import KLT_Tracker
from src.detection.detection import detect_pose, Detector


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
                         corners_3D: np.ndarray) -> np.ndarray:
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
            valid, rvecs, tvec = detect_pose(kpoints_2D, kpoints_3D, cameraMatrix, distCoeffs)
            corners_3D = np.float32(corners_3D)  # 3D coordinates of corners of model
            corners_2D, _ = cv.projectPoints(corners_3D, rvecs, tvec, cameraMatrix,
                                             distCoeffs)  # transform to 2D coordinates using pose
            corners_2D = corners_2D.reshape(-1, 1, 2)
            # frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)

            return corners_2D
        else:
            return None


def track_frame(detector: Detector, video_path: str = None, output_path: str = None, track_length: int = 50,
                fps: int = 30, color: tuple = (255, 0, 0)) -> None:
    '''
    This func tracks object on video (with detection every track_length) and save video to output
    :param detector: Detector, detector, that used to detect object
    :param video_path: str, path to original video
    :param output_path: str, destination where video should be saved
    :param track_length: int, length of frames that should be tracked until new detection
    :param fps: int, frame per second (optional), 30 fps is usually used
    :param color: tuple, color of frame in BGR color scheme (255, 0 , 0) - Blue
    :return: None
    '''
    img = detector.registration_params['img']
    cameraMatrix, distCoeffs = detector.camera_params['mtx'], detector.camera_params['dist']

    cap = cv.VideoCapture(video_path)  # load video file

    tracker = FrameRegistration()

    object_corners_3d = detector.registration_params['object_corners_3d']
    object_corners_2d = detector.registration_params['object_corners_2d']

    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    img_pts, kpoints_3d, kpoints_2d = detector.detect(img)
    mask = np.zeros_like(previous_frame)
    Images = []
    count = 1
    imgSize = None
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if count % track_length == 0:
            img_pts_detected, kpoints_3d_detected, kpoints_2d_detected = detector.detect(previous_frame)

            # handling case of bad detection
            if img_pts_detected is None:
                print("Bad detection of object, replaced by tracking")
                good_new, good_old, kpoints_3d = tracker.track_features_sift(previous_frame, frame, kpoints_2d,
                                                                             kpoints_3d)
                object_corners_2d = tracker.find_new_corners(kpoints_3d, good_new, cameraMatrix, distCoeffs,
                                                             object_corners_3d)
                frame = cv.polylines(frame, [object_corners_2d.astype('int32')], True, color, 3, cv.LINE_AA)
                img = cv.add(frame, mask)
                imgSize = img.shape
                Images.append(img)
                # let tracking work extra 5 frames
                count = track_length - 5
                if len(frame) > 0:
                    previous_frame = frame.copy()
                continue
            img_pts, kpoints_3d, kpoints_2d = img_pts_detected, kpoints_3d_detected, kpoints_2d_detected
            mask = np.zeros_like(previous_frame)

        # track new frame
        good_new, good_old, kpoints_3d = tracker.track_features_sift(previous_frame, frame, kpoints_2d, kpoints_3d)
        object_corners_2d = tracker.find_new_corners(kpoints_3d, good_new, cameraMatrix, distCoeffs, object_corners_3d)
        if object_corners_2d is not None:
            frame = cv.polylines(frame, [np.int32(object_corners_2d)], True, color, 3, cv.LINE_AA)
            img = cv.add(frame, mask)
        else:
            count = -1
            img = frame
        imgSize = img.shape
        Images.append(img)
        if len(frame) > 0:
            previous_frame = frame.copy()

        kpoints_2d = good_new.reshape(-1, 1, 2)

        count += 1

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
    cv.destroyAllWindows()
