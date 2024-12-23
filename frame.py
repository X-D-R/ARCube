
import cv2 as cv
import numpy as np
from registration import Model, register
from detection import Detector








def detect_features(shot, model, h, w, img1):  #detecting features, matching features
        MIN_MATCH_COUNT = 10
        sift = cv.SIFT_create()
        kp1, des1 = model.kp, model.des
        #kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(shot, None)  #detect features
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

        return s_d, features  #3D features, 2D features


def detect_pose(p_feat, sift_3d):
    if len(p_feat) > 3 and len(p_feat) == len(sift_3d):
        valid, rvec, tvec = cv.solvePnP(sift_3d, p_feat, cameraMatrix, distCoeffs)
        rvecs = cv.Rodrigues(rvec)[0]
        return valid, rvecs, tvec
    return False


def registration_transform_3d_corners(rvecs, tvec, cameraMatrix, distCoeffs): #transform 3D coordinates of model corners to 2D
    pts = np.float32(model.object_corners_3d) #3D coordinates of corners of model
    t, _ = cv.projectPoints(pts, rvecs, tvec, cameraMatrix, distCoeffs) #transform to 2D coordinates using pose
    t = t.reshape(-1, 2)
    return t


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

    def registrate_model(self, reference, camera_params, h, w):
        model = Model()
        model.load_camera_params(camera_params)
        model.upload_image(reference, 'output.jpg')
        model.register('SIFT')
        model_path = 'model_params.npz'
        model.crop_method = 'corner'
        model.object_corners_2d = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        model.object_corners_3d = np.array([
                                        [0, 0, 0],  # Top-left
                                        [w-1, 0, 0],  # Top-right
                                        [w-1, h-1, 0],  # Bottom-right
                                        [0, h-1, 0],  # Bottom-left
                                    ], dtype="float32")
        model.save_to_npz(model_path)
        model = Model.load(model_path)
        return model



    def track_features_sift(self, previous_frame, current_frame, feat, sift_3d): #track features KLT
        frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

        p_feat, st_feat, err_feat = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, feat, None, **self.lk_params)

        good_new_feat = p_feat[st_feat == 1]
        good_old_feat = feat[st_feat == 1]

        if len(good_new_feat) != len(sift_3d):   #matching 3D features with 2D features
            new_3d = []
            for i in range(len(st_feat)):
                if st_feat[i] == 1:
                    new_3d.append(sift_3d[i])
            new_3d = np.array(new_3d)
            print(new_3d.shape, good_new_feat.shape)
        else:
            new_3d = sift_3d


        return good_new_feat, good_old_feat, p_feat, new_3d



    def draw_frame(self, frame, sift_3d, sift_2d, cameraMatrix, distCoeffs, model, h, w, p_feat):
        if len(p_feat) > 3 and len(p_feat) == len(sift_3d):
            valid, rvecs, tvec = detect_pose(sift_2d, sift_3d)
            t = registration_transform_3d_corners(rvecs, tvec, cameraMatrix, distCoeffs)
            frame = cv.polylines(frame, [np.int32(t)], True, 255, 3, cv.LINE_AA)

        return frame




# Example usage
if __name__ == "__main__":
    img1 = cv.imread('reference.jpg', cv.IMREAD_GRAYSCALE)
    h, w = img1.shape

    img_ = 'reference.jpg'   #load reference photo for model
    camera_params = 'cam_params.npz'  #load camera parameters

    with np.load('cam_params.npz') as file:
        cameraMatrix = file['cameraMatrix']
        distCoeffs = file['dist']


    cap = cv.VideoCapture('WIN_20241112_20_14_06_Pro.mp4')  #load video file

    tracker = frame_by_registration()
    model = tracker.registrate_model(img_, camera_params, h, w)

    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit()

    pts_3d, feat = detect_features(previous_frame, model, h, w, img1)
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
            pts_3d, feat = detect_features(previous_frame, model, h, w, img1)
            mask = np.zeros_like(previous_frame)
        good_new, good_old, p_feat, pts_3d = tracker.track_features_sift(previous_frame, frame, feat, pts_3d)

        frame = tracker.draw_frame(frame, pts_3d, good_new, cameraMatrix, distCoeffs, model, h, w, p_feat)

        img = cv.add(frame, mask)
        imgSize = img.shape

        Images.append(img)
        if len(frame) > 0:
            previous_frame = frame.copy()
            #print(count)

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