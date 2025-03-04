import cv2 as cv
import numpy as np
from src.registration.rectangle_model import RectangleModel
import time


class Detector:
    def __init__(self, model=None):
        self.MIN_MATCH_COUNT = 10
        self.images: list
        self.registration_params: dict = {}
        self.camera_params: dict = {}
        self.descriptor = cv.SIFT.create()  # base
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=5)
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        self.use_flann = True
        self.previous_rvec = None
        self.previous_tvec = None

    def get_rvec_tvec(self) -> (np.ndarray, np.ndarray):
        return self.previous_rvec, self.previous_tvec

    def instance_method(self, use_flann=True) -> None:
        '''
        Func to instance descriptor and matcher
        :param use_flann: boolean, use Flann-based matcher or not
        :return: None
        '''
        self.use_flann = use_flann
        if self.registration_params['feature_method'] == "ORB":
            self.descriptor = cv.ORB.create()
        elif self.registration_params['feature_method'] == "KAZE":
            self.descriptor = cv.KAZE.create()
        elif self.registration_params['feature_method'] == "AKAZE":
            self.descriptor = cv.AKAZE.create()
        elif self.registration_params['feature_method'] == "BRISK":
            self.descriptor = cv.BRISK.create()
        elif self.registration_params['feature_method'] == "SIFT":
            self.descriptor = cv.SIFT.create()
        else:
            raise ValueError("Unsupported feature type.")

        if self.registration_params['feature_method'] in ["SIFT", "KAZE"]:
            if use_flann:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
                search_params = dict(checks=5)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            else:
                self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self.registration_params['feature_method'] in ["ORB", "AKAZE", "BRISK"]:
            if use_flann:
                print("couldn't use Flann, used Brute Force instead")
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        else:
            self.matcher = cv.BFMatcher()

    def load_camera_params(self, path: str) -> None:
        '''
        This func load camera parameters from
        .npz file
        :param path: str, path to saved parameters as ".npz"
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.camera_params = {
                    "mtx": file['cameraMatrix'],
                    "dist": file['dist']
                }
        else:
            raise ValueError('Error: it is not .npz file')

    def load_model_params(self, path: str) -> None:
        '''
        This func get parameters from file
        saved as ".npz"
        :param path: str, path to saved parameters as ".npz"
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path, allow_pickle=True) as file:
                print(file.get('output_path'), type(file.get('output_path')))
                self.registration_params = {
                    "img": cv.imread(str(file.get('output_path'))),
                    "key_points_2d": file.get('key_points_2d'),
                    "key_points_3d": file.get('key_points_3d'),
                    "des": file.get('des'),
                    "object_corners_2d": file.get('object_corners_2d'),
                    "object_corners_3d": file.get('object_corners_3d'),
                    "feature_method": file.get('feature_method')
                }
        else:
            raise ValueError('Error: it is not .npz file')

    def get_model_params(self, model: RectangleModel) -> None:
        '''
        This func get model parameters from object
        :param model: Model, object of Model class
        :return: None
        '''
        self.registration_params = {
            "img": cv.imread(model.output_path),
            "key_points_2d": model.key_points_2d,
            "key_points_3d": model.key_points_3d,
            "des": model.des,
            "object_corners_2d": model.object_corners_2d,
            "object_corners_3d": model.object_corners_3d,
            "feature_method": model.feature_method
        }

    def set_detector(self, camera_path: str, model_path: str, use_flann: bool = True,
                     camera_params_approximate: dict = {}) -> None:
        if camera_path is None:
            self.camera_params = camera_params_approximate
        else:
            self.load_camera_params(camera_path)
        self.load_model_params(model_path)
        self.instance_method(use_flann)

    def set_detector_by_model(self, camera_path: str, model: RectangleModel, use_flann: bool = True) -> None:
        self.load_camera_params(camera_path)
        self.get_model_params(model)
        self.instance_method(use_flann)

    def detect_path(self, image_path: str, coeff_lowes: int = 0.7) -> (np.ndarray, np.ndarray, np.ndarray, list, list):
        '''
        This func to detect object on image
        :param image_path: str, path to image, there we need to detect object
        :param coeff_lowes: int, 0 < coefficient < 1
        :return: (np.ndarray, np.ndarray, np.ndarray, list, list), result vertices of object, inliers on source image, inliers on that image, good matches, mask for debug
        '''
        img = cv.imread(image_path)
        src_pts, dst_pts = np.float32(), np.float32()
        if img is None:
            raise ValueError("There is no image on this path")
        kp1, des1 = self.registration_params["key_points_3d"], self.registration_params[
            "des"]  # kp should be in 3D real coords
        kp2, des2 = self.descriptor.detectAndCompute(img, None)
        if not self.use_flann:
            matches = self.matcher.match(des1, des2)
        else:
            matches = self.matcher.knnMatch(des1, des2, 2)
        good = self._lowes_ratio_test(matches, coeff_lowes)
        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([[kp1[m.queryIdx][0], kp1[m.queryIdx][1], kp1[m.queryIdx][2]] for m in good]).reshape(
                -1, 1, 3)
            dst_pts = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good]).reshape(-1, 1, 2)
            mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
            valid, rvec, tvec, mask = cv.solvePnPRansac(src_pts, dst_pts, mtx, dist)
            obj_points = self.registration_params["object_corners_3d"]
            rvecs = cv.Rodrigues(rvec)[0]
            img_points, _ = cv.projectPoints(obj_points, rvecs, tvec, mtx, dist)
            inliers_original, inliers_frame = src_pts[mask], dst_pts[mask]

        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            img_points, inliers_original, inliers_frame, good, mask = None, None, None, None, None

        return img_points, inliers_original, inliers_frame, kp2, good

    def detect(self, image: np.ndarray, coeff_lowes: int = 0.5) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        This func to detect object on image
        :param image: np.ndarray, image, there we need to detect object
        :param coeff_lowes: int, 0 < coefficient < 1
        :return: (np.ndarray, np.ndarray, np.ndarray, list, list), result vertices of object, inliers on source image, inliers on that image, good matches, mask for debug
        '''
        src_pts, dst_pts = np.float32(), np.float32()
        if image is None:
            raise ValueError("There is no image on this path")
        kp1, des1 = self.registration_params["key_points_3d"], self.registration_params[
            "des"]  # kp should be in 3D real coords
        kp2, des2 = self.descriptor.detectAndCompute(image, None)
        if not self.use_flann:
            matches = self.matcher.match(des1, des2)
        else:
            matches = self.matcher.knnMatch(des1, des2, 2)
        good = self._lowes_ratio_test(matches, coeff_lowes)
        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([[kp1[m.queryIdx][0], kp1[m.queryIdx][1], kp1[m.queryIdx][2]] for m in good]).reshape(
                -1, 1, 3)
            dst_pts = np.float32([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good]).reshape(-1, 1, 2)
            #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            mtx, dist = self.camera_params["mtx"], self.camera_params["dist"]
            if self.previous_rvec is None or self.previous_tvec is None:
                valid, rvec, tvec, mask = cv.solvePnPRansac(src_pts, dst_pts, mtx, dist, iterationsCount=100,
                                                            reprojectionError=8.0, confidence=0.99,
                                                            flags=cv.SOLVEPNP_ITERATIVE)
                self.previous_rvec, self.previous_tvec = rvec, tvec
            else:
                valid, rvec, tvec, mask = cv.solvePnPRansac(src_pts, dst_pts, mtx, dist, self.previous_rvec,
                                                            self.previous_tvec, useExtrinsicGuess=True,
                                                            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
                                                            flags=cv.SOLVEPNP_ITERATIVE)
                self.previous_rvec, self.previous_tvec = rvec, tvec
            obj_points = self.registration_params["object_corners_3d"]
            if valid:
                rvecs = cv.Rodrigues(rvec)[0]
                img_points, _ = cv.projectPoints(obj_points, rvecs, tvec, mtx, dist)
                mask = mask.ravel().tolist()
                inliers_original, inliers_frame = src_pts[mask], dst_pts[mask]
            else:
                print("No good keypoints")
                img_points, inliers_original, inliers_frame = None, None, None

        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            img_points, inliers_original, inliers_frame = None, None, None

        return img_points, inliers_original, inliers_frame, kp2, good, None, None

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


def detect_pose(p_feat, sift_3d, cameraMatrix, distCoeffs, rvec=None, tvec=None):
    if len(p_feat) > 3 and len(p_feat) == len(sift_3d):
        '''if rvec is not None and tvec is not None:
            valid, rvec, tvec, mask = cv.solvePnPRansac(sift_3d, p_feat, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=True, iterationsCount=100, reprojectionError=8.0, confidence=0.99, flags=cv.SOLVEPNP_ITERATIVE)
        else:
            valid, rvec, tvec, mask = cv.solvePnPRansac(sift_3d, p_feat, cameraMatrix, distCoeffs, iterationsCount=100, reprojectionError=8.0, confidence=0.99, flags=cv.SOLVEPNP_ITERATIVE)'''
        if rvec is not None and tvec is not None:
            valid, rvec, tvec = cv.solvePnP(sift_3d, p_feat, cameraMatrix, distCoeffs, rvec, tvec,
                                                  useExtrinsicGuess=True,
                                                  flags=cv.SOLVEPNP_ITERATIVE)
        else:
            valid, rvec, tvec = cv.solvePnP(sift_3d, p_feat, cameraMatrix, distCoeffs,
                                                  flags=cv.SOLVEPNP_ITERATIVE)
        rvecs = cv.Rodrigues(rvec)[0]
        return valid, rvecs, tvec
    return False
