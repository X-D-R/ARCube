import cv2 as cv
import numpy as np


def draw_tracks(mask, frame, start_keypoints, end_keypoints):
    '''
    Draws lines (tracks) and circles (keypoints) on the frame for visualization of Lucas-Kanade optical flow.

    :param mask: np.ndarray, mask image for drawing purposes.
    :param frame: np.ndarray, video frame or image.
    :param start_keypoints: np.ndarray, array of start 2D keypoints.
    :param end_keypoints: np.ndarray, array of end 2D keypoints.
    :return: (np.ndarray, np.ndarray), updated mask, updated frame.
    '''
    for i, (start, end) in enumerate(zip(start_keypoints, end_keypoints)):
        a, b = start.ravel()
        c, d = end.ravel()
        cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        frame = cv.circle(frame, (int(a), int(b)), 2, (0, 0, 255), -1)

    return mask, frame


def draw_contours_of_rectangle(image_path: str, output_path: str, img_points: np.ndarray, color=(0, 0, 255),
                               thickness=15) -> None:
    '''
    This func draw contours of rectangle
    and save image to selected path
    :param image_path: str, path to source image
    :param output_path: str, path to output image
    :param img_points: np.ndarray, numpy array of 2D points on image, that is 4 points of original rectangle
    :param color: tuple, RGB tuple, for example RED is (255, 0, 0)
    :param thickness: int, the thickness of lines
    '''
    img = cv.imread(image_path)
    img = cv.polylines(img, [np.int32(img_points)], True, color, thickness)
    cv.imwrite(output_path, img)

    max_height = 800
    h, w, channels = img.shape
    if h > max_height:
        scale = max_height / h
        img = cv.resize(img, (int(w * scale), int(h * scale)))
    cv.imshow("Detected photo. Press enter to close", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


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
    img = cv.imread(image_path)
    img = cv.polylines(img, [np.int32(img_points[::2])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[1::2])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[:2:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[2:4:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[4:6:])], True, color, thickness)
    img = cv.polylines(img, [np.int32(img_points[6:8:])], True, color, thickness)
    cv.imwrite(output_path, img)
    return


def split_video_to_frames(video_path: str, output_folder_path: str) -> None:
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


def split_video_to_frames_undistorted(video_path: str, output_folder_path: str, camera_matrix: np.ndarray,
                                      distortion: np.ndarray) -> None:
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

def visualize_matches(reference_image, reference_kp, image, image_kp, matches):
    '''
    Visualizes matches with the reference image.

    :param reference_image: np.ndarray, reference image.
    :param reference_kp: np.ndarray, reference image keypoints.
    :param image: np.ndarray, image for matching.
    :param image_kp: tuple, image keypoints.
    :param matches: list, matches keypoints.
    :return: np.ndarray, image with visualizing of matches with the reference image.
    '''
    keypoints = []
    for kp_dict in reference_kp:
        x = kp_dict.get("pt")[0]
        y = kp_dict.get("pt")[1]
        size = kp_dict.get("size")
        angle = kp_dict.get("angle")
        response = kp_dict.get("response")
        octave = kp_dict.get("octave")
        class_id = kp_dict.get("class_id")
        kp = cv.KeyPoint(x=x, y=y, size=size, angle=angle, response=response, octave=octave, class_id=class_id)
        keypoints.append(kp)

    arr_of_matches = []
    for match in matches:
        arr_of_matches.append([match])

    result = cv.drawMatchesKnn(reference_image, keypoints, image, image_kp, arr_of_matches, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    max_height = 800
    h, w, channels = result.shape
    if h > max_height:
        scale = max_height / h
        result = cv.resize(result, (int(w * scale), int(h * scale)))

    return result

def visualize_matches_on_photo(reference_image, reference_kp, image_path, image_kp, matches):
    '''
    Visualizes matches with the reference image on photo.

    :param reference_image: np.ndarray, reference image.
    :param reference_kp: np.ndarray, reference image keypoints.
    :param image_path: str, path of image.
    :param image_kp: tuple, image keypoints.
    :param matches: list, matches keypoints.
    :return: None
    '''
    img2 = cv.imread(image_path)
    result = visualize_matches(reference_image, reference_kp, img2, image_kp, matches)
    cv.imshow("Visualizing of matches", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
