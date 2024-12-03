import cv2 as cv
import numpy as np

#comment
class KLT_Tracker:
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7, block_size=7):
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

    def detect_features(self, old_frame):
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        return p0

    def find_good(self, img1, shot):
        MIN_MATCH_COUNT = 10
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(shot, None)
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

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            features = []

            ob = []
            for i, m in enumerate(good):
                if matchesMask[i]:
                    features.append(dst_pts[i])
                    ob.append(src_pts[i])

            s_d = []
            for i in ob:
                x, y = i[0][0], i[0][1]
                s_d.append([x, y, 0])
            s_d = np.array(s_d)

            features = np.array(features)

        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        return dst, features, s_d

    def track_features(self, old_frame, frame, p0):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        return good_new, good_old

    def track_features_sift(self, old_frame, frame, feat, flag):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        p_feat, st_feat, err_feat = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, feat, None, **self.lk_params)

        good_new_feat = p_feat[st_feat == 1]
        good_old_feat = feat[st_feat == 1]

        if flag:
            return good_new_feat, good_old_feat, p_feat

        return good_new_feat, good_old_feat

    def draw_tracks(self, mask, frame, good_new, good_old):
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()

            c, d = old.ravel()

            cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)

            frame = cv.circle(frame, (int(a), int(b)), 2, (0, 0, 255), -1)
        return mask, frame

    def draw_frame(self, img1, frame, s_d, p_feat, cameraMatrix, distCoeffs):
        h, w = img1.shape
        pts = np.float32([[0, 0, 0], [0, h - 1, 0], [w - 1, h - 1, 0], [w - 1, 0, 0]])
        if len(p_feat) > 3 and len(p_feat) == len(s_d):
            valid, rvec, tvec = cv.solvePnP(s_d, p_feat, cameraMatrix, distCoeffs)
            rvecs = cv.Rodrigues(rvec)[0]

            t, _ = cv.projectPoints(pts, rvecs, tvec, cameraMatrix, distCoeffs)

            t = t.reshape(-1, 2)
            frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)
        return frame


# Example usage
if __name__ == "__main__":
    img1 = cv.imread('reference.jpg', cv.IMREAD_GRAYSCALE)
    cameraMatrix = np.loadtxt('cameraMatrix.txt')
    distCoeffs = np.loadtxt('distCoeffs.txt')
    cap = cv.VideoCapture('WIN_20241112_20_14_06_Pro.mp4')
    klt_tracker = KLT_Tracker()

    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    #p0 = klt_tracker.detect_features(old_frame)
    cor, feat, s_d = klt_tracker.find_good(img1, old_frame)
    mask = np.zeros_like(old_frame)
    Images = []
    n = 100
    count = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if count % n == 0:
            ret, old_frame = cap.read()
            #p0 = klt_tracker.detect_features(old_frame)
            cor, feat, s_d = klt_tracker.find_good(img1, old_frame)
            mask = np.zeros_like(old_frame)


        #good_new, good_old = klt_tracker.track_features(old_frame, frame, p0)
        good_new, good_old, p_feat = klt_tracker.track_features_sift(old_frame, frame, feat, 1)

        good_new_cor, good_old_cor = klt_tracker.track_features_sift(old_frame, frame, cor, 0)

        #mask, frame = klt_tracker.draw_tracks(mask, frame, good_new, good_old)
        mask, frame = klt_tracker.draw_tracks(mask, frame, good_new, good_old)
        mask, frame = klt_tracker.draw_tracks(mask, frame, good_new_cor, good_old_cor)
        frame = klt_tracker.draw_frame(img1, frame, s_d, p_feat, cameraMatrix, distCoeffs)

        img = cv.add(frame, mask)
        imgSize = img.shape

        Images.append(img)

        old_frame = frame.copy()

        #p0 = good_new.reshape(-1, 1, 2)
        feat = good_new.reshape(-1, 1, 2)
        cor = good_new_cor.reshape(-1, 1, 2)

        count += 1

    height, width, channels = imgSize

    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    video = cv.VideoWriter('sifts.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
