
import cv2 as cv
import numpy as np
from registration import Model, register
from detection import Detector




class frame_by_registration:
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

    def registrate_model(self, reference, camera_params):
        model = Model()
        model.load_camera_params(camera_params)

        model.upload_image(reference, 'output.jpg')
        model.register('SIFT')
        model_path = 'model_params.npz'
        model.save_to_npz(model_path)
        model = Model.load(model_path)
        return model


    def find_good_sift(self, shot, model):
        MIN_MATCH_COUNT = 10
        kp1, des1 = model.kp, model.des
        detector = Detector()
        detector.get_model_params(model)
        detector.instance_method(True)

        detector.detect_image(shot, useFlann=True, drawMatch=False)
        kp2, des2 = detector.kp, detector.des
        sift_3d = detector.p_3D
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


            for i, m in enumerate(good):
                if matchesMask[i]:
                    features.append(dst_pts[i])


            features = np.array(features)

        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        '''return dst, features, s_d'''
        return sift_3d, features


    def track_features_sift(self, previous_frame, current_frame, feat, flag):
        frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

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

    def draw_frame(self, frame, sift_3d, p_feat, cameraMatrix, distCoeffs, model):
        #h, w = img1.shape
        #pts = np.float32([[0, 0, 0], [0, h - 1, 0], [w - 1, h - 1, 0], [w - 1, 0, 0]])
        pts = model.corners
        if len(p_feat) > 3 and len(p_feat) == len(sift_3d):
            valid, rvec, tvec = cv.solvePnP(sift_3d, p_feat, cameraMatrix, distCoeffs)
            rvecs = cv.Rodrigues(rvec)[0]

            t, _ = cv.projectPoints(pts, rvecs, tvec, cameraMatrix, distCoeffs)

            t = t.reshape(-1, 2)
            frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)
        return frame


# Example usage
if __name__ == "__main__":
    img1 = 'reference.jpg'   #load reference photo

    camera_params = 'cam_params.npz'  #load camera parameters

    with np.load('cam_params.npz') as file:
        cameraMatrix = file['cameraMatrix']
        distCoeffs = file['dist']


    cap = cv.VideoCapture('WIN_20241112_20_14_06_Pro.mp4')  #load video file

    tracker = frame_by_registration()
    model = tracker.registrate_model(img1, camera_params)

    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    pts_3d, feat = tracker.find_good_sift(previous_frame, model)
    mask = np.zeros_like(previous_frame)
    Images = []
    n = 100
    count = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if count % n == 0:
            ret, previous_frame = cap.read()
            pts_3d, feat = tracker.find_good_sift(previous_frame, model)
            mask = np.zeros_like(previous_frame)


        good_new, good_old, p_feat = tracker.track_features_sift(previous_frame, frame, feat, 1)


        frame = tracker.draw_frame(frame, pts_3d, p_feat, cameraMatrix, distCoeffs, model)

        img = cv.add(frame, mask)
        imgSize = img.shape

        Images.append(img)

        previous_frame = frame.copy()

        feat = good_new.reshape(-1, 1, 2)


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