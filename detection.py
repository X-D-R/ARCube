import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from registration import Model


class Detector():

    def __init__(self, model=None):
        self.MIN_MATCH_COUNT = 10
        self.images: list
        self.registration_params: dict = {}
        self.camera_params: dict = {}
        self.descriptor = cv.ORB.create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def _trans(self, points_r, rvecs, tvec, cam_m, dist):
        p_2d = np.zeros((len(points_r), 2))
        dist = np.pad(dist[0], (0, 6 - len(dist[0])), 'constant', constant_values=0)
        for i in range(len(points_r)):
            p_t = (rvecs @ points_r[i] + tvec.T)[0]
            x_ = p_t[0] / p_t[2]
            y_ = p_t[1] / p_t[2]
            r_2 = np.power(x_, 2) + np.power(y_, 2)
            coeff = (1 + dist[0] * r_2 + dist[1] * np.power(r_2, 2) + dist[2] * np.power(r_2, 3)) / (
                        1 + dist[3] * r_2 + dist[4] * np.power(r_2, 2) + dist[5] * np.power(r_2, 3))
            u = cam_m[0][0] * x_ * coeff + cam_m[0][2]
            v = cam_m[1][1] * y_ * coeff + cam_m[1][2]
            p_2d[i][0] = u
            p_2d[i][1] = v

        return p_2d


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        path should be .npz file
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.camera_params = {
                    "mtx": file['cameraMatrix'],
                    "dist": file['dist'],
                    "rvecs": file['rvecs'],
                    "tvecs": file['tvecs']
                }
        else:
            print('Error: it is not .npz file')
    def instance_method(self, useFlann=True) -> None:
        '''
        Func to instance descriptor and matcher
        :param useFlann: boolean
        :return:
        '''
        if self.registration_params['method'] == "ORB":
            self.descriptor = cv.ORB.create()
        elif self.registration_params['method'] == "KAZE":
            self.descriptor = cv.KAZE.create()
        elif self.registration_params['method'] == "AKAZE":
            self.descriptor = cv.AKAZE.create()
        elif self.registration_params['method'] == "BRISK":
            self.descriptor = cv.BRISK.create()
        elif self.registration_params['method'] == "SIFT":
            self.descriptor = cv.SIFT.create()
        else:
            raise ValueError("Unsupported feature type.")

        if self.registration_params['method'] in ["SIFT", "KAZE"]:
            if useFlann:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            else:
                self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self.registration_params['method'] in ["ORB", "AKAZE", "BRISK"]:
            if useFlann:
                print("cant use Flann")
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        else:
            self.matcher = cv.BFMatcher()


    def load_model_params(self, path) -> None:
        '''
        This function should load kp and des
        from file that was created with model.save_to_npz
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.registration_params = {
                    "img": cv.imread(file[''], cv.IMREAD_GRAYSCALE),
                    "height": file['height'],
                    "width": file['width'],
                    "kp": file['kp'],
                    "des": file['des'],
                    "vol": file['vol'],
                    "camera_params": file['camera_params'],
                    "method": file['method']
                }
            self.camera_params = file['camera_params']
        else:
            print('Error: it is not .npz file')


    def get_model_params(self, model: Model) -> None:
        '''
        This function should load kp and des
        from file that was created with model.save_to_npz
        :return: None
        '''
        self.registration_params = {
            "img": model.img,
            "height": model.height,
            "width": model.width,
            "kp": model.kp,
            "des": model.des,
            "vol": model.vol,
            "camera_params": model.camera_params,
            "method": model.method
        }
        self.camera_params = model.camera_params


    def detect_video(self, path) -> None:
        '''
        This function should detect object in video and
        draw box. Then save to self.detected_images
        using self.detect_image
        :return: None
        '''
        self._upload_video(path)
        ind = 0
        image = cv.imread('./videoframes/frame_0.png')
        height, width = image.shape[:2]
        out = cv.VideoWriter('output_video.mp4', cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 5.0, (width, height))
        while ind < 90:
            print(ind)
            frame = self.detect_image('./videoframes/frame_' + str(ind) + '.png', drawMatch=False)
            if frame is None:
                break
            #plt.imshow(frame, 'gray'), plt.show()
            ind += 1
            out.write(frame)

        out.release()

    def detect_image(self, path, useFlann=True, drawMatch=False):
        '''
        This function should detect object on image and
        draw box. Then save to self.detected_images using
        self.draw_box
        :return: None
        '''
        img_colored = cv.imread(path)
        img2 = self._upload_image(path)
        imgRes = img2
        if img2 is None:
            return None
        kp1, des1 = self.registration_params["kp"], self.registration_params["des"]
        kp2, des2 = self.descriptor.detectAndCompute(img2, None)
        matches = self.matcher.knnMatch(des1, des2, 2)
        good = self._lowes_ratio_test(matches, 0.6)
        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            h, w = self.registration_params["height"], self.registration_params["width"]
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w - 1, w - 1], [0, w - 1]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            #imgRes = cv.polylines(img_colored, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            imgRes = self._draw_box(img_colored, dst, dst_pts, good, matchesMask)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            matchesMask = None

        if drawMatch:
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img1 = self.registration_params["img"]
            img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            plt.imshow(img3, 'gray'), plt.show()

        return imgRes


    def _draw_box(self, img, dst, dst_pts, good, matchesMask):
        '''
        This function should draw box of detected
        object using given points
        Order of points: (write)
        :return: None
        '''
        img2 = cv.polylines(img, [np.int32(dst[:4, :, :])], True, (0, 255, 255), 6)
        h2, w2 = img2.shape[:2]
        w, h, z = self.registration_params["width"], self.registration_params["height"], self.registration_params["vol"]
        mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w2, h2), 1, (w2, h2))
        objPoints = np.array([[0., 0., 0.], [w - 1, 0., 0.], [w - 1, w - 1, 0.], [0., w - 1, 0.]])
        valid, rvec, tvec = cv.solvePnP(objPoints, dst[[True, True, False, False, True, True]], newcameramtx, dist, cv.SOLVEPNP_P3P)
        # cv.drawFrameAxes(img2, newcameramtx, dist, rvec, tvec, 10000)
        rvecs = cv.Rodrigues(rvec)[0]
        objPoints = np.array(
            [[0., 0., 0.], [0., 0., z - 1], [w - 1, 0., 0.], [w - 1, 0., z - 1], [w - 1, h - 1, 0.],
             [w - 1, h - 1, z - 1],
             [0., h - 1, 0.], [0., h - 1, z - 1]])
        rvecs = cv.Rodrigues(rvec)[0]
        d2 = self._trans(objPoints, rvecs, tvec, newcameramtx, dist)
        # img2 = cv.polylines(imgS,[np.int32(d2)],True, (0, 0, 255), 2)
        diff_z = np.array(d2[1::2] - d2[::2])
        new_dst = dst[:4, :, :] + diff_z.reshape(-1, 1, 2)
        img2 = cv.polylines(img2, [np.int32(new_dst)], True, (0, 255, 255), 6)
        col_dst = [dst[0], new_dst[0], new_dst[1], dst[1], dst[2], new_dst[2], new_dst[3], dst[3]]
        img2 = cv.polylines(img2, [np.int32(col_dst)], True, (0, 255, 255), 6)
        for i, m in enumerate(good):
            pt = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
            if matchesMask[i]:
                cv.circle(img2, pt, 5, (0, 255, 255), -1)  # Inliers желтые
            else:
                cv.circle(img2, pt, 5, (0, 0, 255), -1)  # Outliers красные

        return img2

    def _upload_image(self, path: str) -> np.ndarray:
        '''
        This func should upload image using cv
        from given path and return it
        :param path: str
        :return: np.ndarray
        '''
        return cv.imread(path, cv.IMREAD_GRAYSCALE)

    def _upload_video(self, path: str) -> None:
        '''
        This func should upload video using cv
        from given path and save as array of images
        :param path: str
        :return: None
        '''
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            print("Error opening video file")

        ind = 0
        mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame is None:
                break
            h, w = frame.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # undistort
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            cv.imwrite('./videoframes/frame_' + str(ind) + '.png', frame)
            ind += 1


    def _lowes_ratio_test(self, matches, coeff=0.7) -> list:
        good = []
        for m, n in matches:
            if m.distance < coeff * n.distance:
                good.append(m)

        return good


def detect(args):
    '''
    Detect features in an image or video.
    params:
        args : The arguments parsed from the command line. Expected arguments include:
              - model_input: Path to the saved model file.
              - camera_params: Path to the camera parameters file (optional for detection).
              - input_image: Path to the input image for detection (optional).
              - input_video: Path to the input video for detection (optional).
              - use_flann: Flag to use FLANN-based matching (optional).
              - draw_match: Flag to draw matches on the detected image (optional).
    '''
    model = Model.load(args.model_input)
    detector = Detector()
    detector.get_model_params(model)

    if args.camera_params:
        detector.load_camera_params(args.camera_params)

    detector.instance_method(args.use_flann)

    if args.input_image:
        detector.detect_image(args.input_image, useFlann=args.use_flann, drawMatch=args.draw_match)
    elif args.input_video:
        detector.detect_video(args.input_video)
    else:
        print("No input image or video provided for detection.")