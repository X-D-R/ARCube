import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from registration import Model


class Detector:
    def __init__(self, model=None):
        self.MIN_MATCH_COUNT = 10
        self.images: list
        self.registration_params: dict = {}
        self.camera_params: dict = {}
        self.descriptor = cv.ORB.create()  # base
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # base

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
                print("couldn't use Flann, used Brute Force instead")
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
                    "img": file['img'],
                    "height": file['height'],
                    "width": file['width'],
                    "kp": file['kp'],
                    "des": file['des'],
                    "vol": file['vol'],
                    "camera_params": file['camera_params'],
                    "method": file['method']
                }
        else:
            raise ValueError('Error: it is not .npz file')

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

    # old func for drawing frame in video
    # def _detect_video(self, video_path, output_path) -> None:
    #     '''
    #     This function should detect object in video and
    #     draw box. Then save to self.detected_images
    #     using self.detect_image
    #     :return: None
    #     '''
    #     self._upload_video(video_path)
    #     ind = 0
    #     image = cv.imread('./videoframes/frame_0.png')
    #     height, width = image.shape[:2]
    #     out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30.0, (width, height))
    #     while True:
    #         print(ind)
    #         frame = self.detect_image('./videoframes/frame_' + str(ind) + '.png', drawMatch=False)
    #         if frame is None:
    #             break
    #         #plt.imshow(frame, 'gray'), plt.show()
    #         ind += 1
    #         out.write(frame)
    #
    #     out.release()

    def detect(self, image_path: str, coeff_lowes=0.7, useFlann=True) -> (np.ndarray, np.ndarray, np.ndarray):
        img = self._upload_image(image_path)
        if img is None:
            raise ValueError("There is no image on this path")
        kp1, des1 = self.registration_params["kp"], self.registration_params["des"]  # kp should be in 3D real coords
        kp2, des2 = self.descriptor.detectAndCompute(img, None)
        h, w, z = self.registration_params["height"], self.registration_params["width"], self.registration_params["vol"]
        if not useFlann:
            matches = self.matcher.match(des1, des2)
        else:
            matches = self.matcher.knnMatch(des1, des2, 2)
        good = self._lowes_ratio_test(matches, coeff_lowes)
        if len(good) > self.MIN_MATCH_COUNT:
            #src_pts = np.float32([[kp1[m.queryIdx].pt[0] * 0.26/w, kp1[m.queryIdx].pt[1] * 0.185/h, 0] for m in good]).reshape(-1, 1, 3)
            src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 3)
            dst_pts = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good]).reshape(-1, 1, 2)
            mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
            valid, rvec, tvec, mask = cv.solvePnPRansac(src_pts, dst_pts, mtx, dist)
            #w, h, z = 0.26, 0.185, 0.01
            obj_points = np.array(
                [[0., 0., 0.], [0., 0., z], [w, 0., 0.], [w, 0., z], [w, h, 0.],
                 [w, h, z],
                 [0., h, 0.], [0., h, z]])
            rvecs = cv.Rodrigues(rvec)[0]
            img_points, _ = cv.projectPoints(obj_points, rvecs, tvec, mtx, dist)
            inliers_original, inliers_frame = src_pts[mask], dst_pts[mask]

        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            img_points, inliers_original, inliers_frame = None, None, None

        return img_points, inliers_original, inliers_frame

    def _draw_frame(self, image_path: str, output_path: str, img_points: np.ndarray, color=(0, 0, 255),
                    thickness=15) -> None:
        img = self._upload_image(image_path)
        img = cv.polylines(img, [np.int32(img_points[::2])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[1::2])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[:2:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[2:4:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[4:6:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[6:8:])], True, color, thickness)
        cv.imwrite(output_path, img)
        return

    def _upload_image(self, path: str) -> np.ndarray:
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

    def _upload_video(self, path: str) -> None:
        '''
        This func should upload video using cv
        from given path and save as array of images
        :param path: str
        :return: None
        '''
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

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
