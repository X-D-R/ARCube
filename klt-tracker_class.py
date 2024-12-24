import cv2 as cv
import numpy as np
from draw_functions import draw_tracks

class KLT_Tracker:
    def __init__(self, max_corners=300, quality_level=0.1, min_distance=7, block_size=4):
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
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        points = cv.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        return points

    def track(self, previous_frame, current_frame, points):
        current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        previous_frame_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

        new_points, status, errors = cv.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, points, None, **self.lk_params)

        good_new_points = new_points[status == 1]
        good_old_points = points[status == 1]

        return good_new_points, good_old_points, status


# Example usage
if __name__ == "__main__":
    cap = cv.VideoCapture('your_video.mp4')
    klt_tracker = KLT_Tracker()

    ret, frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    points = klt_tracker.get_points_to_track(frame)
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
            points = klt_tracker.get_points_to_track(frame)
            mask = np.zeros_like(frame)

        good_new_points, good_old_points, status = klt_tracker.track(frame, current_frame, points)
        mask, current_frame = draw_tracks(mask, current_frame, good_new_points, good_old_points)

        img = cv.add(current_frame, mask)
        imgSize = img.shape
        Images.append(img)
        frame = current_frame.copy()
        points = good_new_points.reshape(-1, 1, 2)
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
