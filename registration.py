import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Model():


    def __init__(self):
        self.img: np.ndarray = np.array(1)
        self.kp: tuple = tuple()
        self.des: np.ndarray = np.array(1)
        self.camera_params: dict = {}


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        path should be .npz file
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]
                self.camera_params["mtx"] = mtx
                self.camera_params["dist"] = dist
                self.camera_params["rvecs"] = rvecs
                self.camera_params["tvecs"] = tvecs


    def upload_image(self, path: str) -> None:
        '''
        This func should upload image using cv
        from given path and save to self.img
        :param path: str
        :return: None
        '''
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)


    def register(self, feature: str) -> None:
        '''
        This function should register model and
        write keypoints and descriptors in self.kp and self.des
        :return: None
        '''
        if feature == "ORB":
            orb = cv.ORB.create()
            self.kp, self.des = orb.detectAndCompute(self.img, None)
        elif feature == "KAZE":
            kaze = cv.KAZE.create()
            self.kp, self.des = kaze.detectAndCompute(self.img, None)
        elif feature == "AKAZE":
            akaze = cv.AKAZE.create()
            self.kp, self.des = akaze.detectAndCompute(self.img, None)
        elif feature == "BRISK":
            brisk = cv.BRISK.create()
            self.kp, self.des = brisk.detectAndCompute(self.img, None)
        elif feature == "SIFT":
            sift = cv.SIFT.create()
            self.kp, self.des = sift.detectAndCompute(self.img, None)


    def _crop_image(self, img: np.ndarray, points: np.ndarray) -> None:
        '''
        This function should crop image
        by given points
        :param img: np.ndarray
        :return: None
        '''
        pass


    def _check(self, path_params: str, path_img: str) -> None:
        self.load_camera_params(path_params)
        self.upload_image(path_img)
        for feature in ["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"]:
            self.register(feature)
            print(f"Feature: {feature}\n\n")
            print(f" KeyPoints: \n {self.kp} \n\n Descriptors: \n{self.des}\n\n")


    def save_to_npz(self) -> None:
        np.savez("RegisterParams", kp=self.kp, des=self.des)


#model = Model()
#model._check("./CameraParams/CameraParams.npz", "./old_files/DanielFiles/book.jpg")