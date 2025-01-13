import cv2 as cv
import numpy as np
from src.utils.draw_functions import draw_tracks


class KLT_Tracker:
    '''
    Class for tracking features using the Kanade-Lucas-Tomasi (KLT) algorithm.
    '''

    def __init__(self, max_corners=300, quality_level=0.1, min_distance=7, block_size=4):
        '''
        Initializes the KLT_Tracker class with the provided parameters.

        :param max_corners: int, maximum number of corners to return (default to 300).
        :param quality_level: float, parameter characterizing the minimal accepted quality of image corners (default to 0.1).
        :param min_distance: float, minimum possible Euclidean distance between the returned corners (default to 7).
        :param block_size: int, size of an average block for computing a derivative covariation matrix over each pixel neighborhood (default to 4).
        '''
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=self.max_corners,
                                   qualityLevel=self.quality_level,
                                   minDistance=self.min_distance,
                                   blockSize=self.block_size)

    def get_points_to_track(self, frame):
        '''
        Detects keypoints in the frame using cv.goodFeaturesToTrack() function.

        :param frame: np.ndarray, video frame or image.
        :return: np.ndarray, array of 2D keypoints detected in the frame.
        '''
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        keypoints = cv.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        return keypoints

    def track(self, previous_frame, current_frame, keypoints):
        '''
        Tracks keypoints using Lucas-Kanade optical flow.

        :param previous_frame: np.ndarray, video frame or image at time (t).
        :param current_frame: np.ndarray, video frame or image at time (t+1).
        :param keypoints: np.ndarray, array of 2D keypoints from previous_frame.
        :return: (np.ndarray, np.ndarray, np.ndarray), array of good 2D keypoints from current_frame,
                 array of good 2D keypoints from previous_frame, status array where each element is set to 1
                 if the flow for the corresponding features has been found, otherwise, it is set to 0.
        '''
        current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        previous_frame_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

        new_keypoints, status, errors = cv.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, keypoints,
                                                                None, **self.lk_params)

        good_new_keypoints = new_keypoints[status == 1]
        good_old_keypoints = keypoints[status == 1]

        return good_new_keypoints, good_old_keypoints, status


# Example usage
if __name__ == "__main__":
    cap = cv.VideoCapture('your_video.mp4')
    klt_tracker = KLT_Tracker()

    ret, frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    keypoints = klt_tracker.get_points_to_track(frame)
    mask = np.zeros_like(frame)
    Images = []
    n = 100
    count = 1
    while True:
        ret, current_frame = cap.read()
        if current_frame is None:
            break

        if count % n == 0:
            ret, frame = cap.read()
            keypoints = klt_tracker.get_points_to_track(frame)
            mask = np.zeros_like(frame)

        good_new_keypoints, good_old_keypoints, status = klt_tracker.track(frame, current_frame, keypoints)
        mask, current_frame = draw_tracks(mask, current_frame, good_new_keypoints, good_old_keypoints)

        img = cv.add(current_frame, mask)
        imgSize = img.shape
        Images.append(img)
        frame = current_frame.copy()
        keypoints = good_new_keypoints.reshape(-1, 1, 2)
        count += 1

    height, width, channels = imgSize
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter('result.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
