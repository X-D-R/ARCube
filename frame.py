import cv2 as cv
import numpy as np
from rectangle_model import register, RectangleModel
from klt_tracker_class import KLT_Tracker
from detection import detect_pose, Detector


def detect_features(shot, model, h, w, img1):  # detecting features, matching features
    MIN_MATCH_COUNT = 10
    sift = cv.SIFT_create()
    kp1, des1 = model.kp, model.des
    # kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(shot, None)  # detect features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        features = []

        ob = []
        for i, m in enumerate(good):
            if matchesMask[i]:
                features.append(dst_pts[i])
                ob.append(src_pts[i])

        s_d = []
        for i in ob:
            x, y = i[0][0], i[0][1]
            s_d.append([[x, y, 0]])
        s_d = np.array(s_d)

        features = np.array(features)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    return s_d, features  # 3D features, 2D features


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
            #frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)

            return corners_2D
        else:
            return None


# Example usage


def old_track_frame(reference_path: str = None, camera_parameters_path: str = None, video_path: str = None) -> None:
    img1 = cv.imread(reference_path, cv.IMREAD_GRAYSCALE)
    h, w = img1.shape

    #img_ = 'reference.jpg'  # load reference photo for model
    #camera_params = 'cam_params.npz'  # load camera parameters

    with np.load(camera_parameters_path) as file:
        cameraMatrix = file['cameraMatrix']
        distCoeffs = file['dist']

    cap = cv.VideoCapture(video_path)  # load video file

    tracker = FrameRegistration()

    object_corners_3d = np.array([
        [0, 0, 0],  # Top-left
        [w - 1, 0, 0],  # Top-right
        [w - 1, h - 1, 0],  # Bottom-right
        [0, h - 1, 0],  # Bottom-left
    ], dtype="float32")
    model = register(
        input_image=reference_path,
        output_image="output_script_test.jpg",
        object_corners_3d=object_corners_3d,
        crop_method='corner',
        feature_method="SIFT",
        model_output="model_script_test.npz"
    )
    with np.load("model_script_test.npz") as file:
        corners2D = file['2d']
        corners3D = file['3d']

    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    kpoints_3d, kpoints_2d = detect_features(previous_frame, model, h, w, img1)
    mask = np.zeros_like(previous_frame)
    Images = []
    n = 50
    count = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if count % n == 0:
            ret, previous_frame = cap.read()
            kpoints_3d, kpoints_2d = detect_features(previous_frame, model, h, w, img1)
            mask = np.zeros_like(previous_frame)
        good_new, good_old, kpoints_3d = tracker.track_features_sift(previous_frame, frame, kpoints_2d, kpoints_3d)
        corners_2D = FrameRegistration.find_new_corners(kpoints_3d, kpoints_2d, cameraMatrix, distCoeffs, corners3D)
        frame = cv.polylines(frame, [np.int32(corners_2D)], True, 255, 3, cv.LINE_AA)
        img = cv.add(frame, mask)
        imgSize = img.shape

        Images.append(img)
        if len(frame) > 0:
            previous_frame = frame.copy()
            # print(count)

        kpoints_2d = good_new.reshape(-1, 1, 2)

        count += 1
    #saving video
    height, width, channels = imgSize

    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    video = cv.VideoWriter('sifts.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


def track_frame(detector: Detector, video_path: str = None, output_path: str = None, track_length: int = 50) -> None:
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
    n = track_length
    count = 1
    imgSize = None
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if count % n == 0:
            img_pts, kpoints_3d, kpoints_2d = detector.detect(previous_frame)
            if img_pts is None:
                Images.append(frame)
                print("No detected image")
                if len(frame) > 0:
                    previous_frame = frame.copy()
                continue
            mask = np.zeros_like(previous_frame)
        good_new, good_old, kpoints_3d = tracker.track_features_sift(previous_frame, frame, kpoints_2d, kpoints_3d)
        object_corners_2d = tracker.find_new_corners(kpoints_3d, good_new, cameraMatrix, distCoeffs, object_corners_3d)
        frame = cv.polylines(frame, [np.int32(object_corners_2d)], True, 255, 3, cv.LINE_AA)
        img = cv.add(frame, mask)
        imgSize = img.shape

        Images.append(img)
        if len(frame) > 0:
            previous_frame = frame.copy()

        kpoints_2d = good_new.reshape(-1, 1, 2)

        count += 1
    #saving video
    height, width, channels = imgSize

    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    video = cv.VideoWriter(output_path, fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    video.release()
    print("Saved video")
    cv.destroyAllWindows()
