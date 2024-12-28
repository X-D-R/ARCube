import cv2 as cv
import numpy as np


def draw_tracks(mask, frame, good_new, good_old):
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        frame = cv.circle(frame, (int(a), int(b)), 2, (0, 0, 255), -1)

    return mask, frame


def upload_image(path: str) -> np.ndarray:
    '''
    This func should upload image using cv
    from given path and return it
    :param path: str
    :return: np.ndarray
    '''
    img = cv.imread(path)
    if img is None:
        raise ValueError("Error opening image file")
    return img


def draw_contours_of_box(image_path: str, output_path: str, img_points: np.ndarray, color=(0, 0, 255),
                         thickness=15) -> None:
    '''
    This func draw contours of box
    and save image to selected path
    :param image_path: str, path to source image
    :param output_path: str, path to output image
    :param img_points: np.ndarray, numpy array of 2D points on image, that is 8 points of original box
    :param color: tuple, RGB tuple, for example RED is (255, 0, 0)
    :param thickness: int, the thickness of lines
    '''
    img = upload_image(image_path)
    img = cv.polylines(img, [np.int32(img_points[::2])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[1::2])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[:2:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[2:4:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[4:6:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[6:8:])], True, color, thickness)
    cv.imwrite(output_path, img)
    return


def upload_video_by_frames(video_path: str, output_folder_path: str) -> None:
    '''
    This func should upload video using cv
    from given path and save as array of images
    :param video_path: str, video that uploaded
    :param output_folder_path: str, folder to which frames should be saved
    :return: None
    '''
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    ind = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        h, w = frame.shape[:2]
        cv.imwrite(output_folder_path + 'frame_' + str(ind) + '.png', frame)
        ind += 1


def upload_video_by_frames_undistorted(video_path: str, output_folder_path: str, camera_matrix: np.ndarray, distortion: np.ndarray) -> None:
    '''
    This func upload video using cv
    from given path and save as undistorted images to chosen folder
    :param video_path: str, video that uploaded
    :param output_folder_path: str, folder to which frames should be saved
    :param camera_matrix: np.ndarray, camera matrix
    :param distortion: np.ndarray, distortion coefficients of camera
    :return: None
    '''
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    ind = 0
    mtx, dist = camera_matrix, distortion
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
        cv.imwrite(output_folder_path + 'frame_' + str(ind) + '.png', frame)
        ind += 1

