import cv2 as cv
import numpy as np
from registration import Model
from Draw_functions import upload_image


class Detector:
    def __init__(self, model=None):
        self.MIN_MATCH_COUNT = 10
        self.images: list
        self.registration_params: dict = {}
        self.camera_params: dict = {}
        self.descriptor = cv.ORB.create()  # base
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # base

    def instance_method(self, useFlann=True) -> None:
        '''
        Func to instance descriptor and matcher
        :param useFlann: boolean, use Flann-based matcher or not
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
        This func get parameters from file
        saved as ".npz"
        :param path: str, path to saved parameters as ".npz"
        :return: np.ndarray
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
        This func get model parameters from object
        :param model: Model, object of Model class
        :return: np.ndarray
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

    def detect(self, image_path: str, coeff_lowes: int=0.7, useFlann: bool=True) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        This func to detect object on image
        :param image_path: str, path to image, there we need to detect object
        :param coeff_lowes: int, 0 < coefficient < 1
        :param useFlann: bool, use Flann-based matcher or not
        :return: (np.ndarray, np.ndarray, np.ndarray), result vertices of object, inliers on source image, inliers on that image
        '''
        img = upload_image(image_path)
        if img is None:
            raise ValueError("There is no image on this path")
        kp1, des1 = self.registration_params["kp"], self.registration_params["des"]  # kp should be in 3D real coords
        kp2, des2 = self.descriptor.detectAndCompute(img, None)
        # h, w, z should be the length width and volume of object in metric system
        h, w, z = self.registration_params["height"], self.registration_params["width"], self.registration_params["vol"]
        if not useFlann:
            matches = self.matcher.match(des1, des2)
        else:
            matches = self.matcher.knnMatch(des1, des2, 2)
        good = self._lowes_ratio_test(matches, coeff_lowes)
        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 0] for m in good]).reshape(-1, 1, 3)
            #src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 3)
            dst_pts = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good]).reshape(-1, 1, 2)
            mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
            valid, rvec, tvec, mask = cv.solvePnPRansac(src_pts, dst_pts, mtx, dist)
            #w, h, z = 0.26, 0.185, 0.01
            # obj_points = np.array(
            #     [[0., 0., 0.], [0., 0., z], [w, 0., 0.], [w, 0., z], [w, h, 0.],
            #      [w, h, z],
            #      [0., h, 0.], [0., h, z]])
            obj_points = self.registration_params["object_corners_3d"]
            rvecs = cv.Rodrigues(rvec)[0]
            img_points, _ = cv.projectPoints(obj_points, rvecs, tvec, mtx, dist)
            inliers_original, inliers_frame = src_pts[mask], dst_pts[mask]

        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            img_points, inliers_original, inliers_frame = None, None, None

        return img_points, inliers_original, inliers_frame

    def _lowes_ratio_test(self, matches, coefficient=0.7) -> list:
        '''
        This func test symmetry of matches
        :param matches: np.ndarray, matches of key points
        :param coefficient: int, 0 < coefficient < 1
        :return: np.ndarray, good matches
        '''
        good = []
        for m, n in matches:
            if m.distance < coefficient * n.distance:
                good.append(m)

        return good
