import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from registration import Model

class Detector():


    def __init__(self, model=None):
        self.images: np.ndarray((-1, 2))
        self.kp: list = []
        self.des: np.ndarray = np.empty((0, 0))
        self.camera_params: dict = {}


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]
                self.camera_params["mtx"] = mtx
                self.camera_params["dist"] = dist
                self.camera_params["rvecs"] = rvecs
                self.camera_params["tvecs"] = tvecs


    def load_model_params(self, path) -> None:
        '''
        This function should load kp and des
        from file that was created with model.save_to_npz
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.kp, self.des = [file[i] for i in ('RegisterParams', 'kp', 'des')]



    def detect_video(self) -> None:
        '''
        This function should detect object in video and
        draw box. Then save to self.detected_images
        using self.detect_image
        :return: None
        '''
        pass

    def detect_image(self) -> None:
        '''
        This function should detect object on image and
        draw box. Then save to self.detected_images using
        self.draw_box
        :return: None
        '''
        img = cv.imread(fname)
        pass


    def draw_box(self, points) -> None:
        '''
        This function should draw box of detected
        object using given points
        Order of points: (write)
        :return: None
        '''
        pass

    def upload_image(self, path: str) -> None:
        '''
        This func should upload image using cv
        from given path and save to images
        :param path: str
        :return: None
        '''
        pass

    def upload_video(self, path: str) -> None:
        '''
        This func should upload video using cv
        from given path and save as array of images
        :param path: str
        :return: None
        '''
        pass