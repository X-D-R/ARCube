import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Model():


    def __init__(self):
        self.img: np.ndarray = np.array(1)
        self.kp: tuple = tuple()
        self.des: np.ndarray = np.array(1)


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        :return: None
        '''
        pass


    def register(self) -> None:
        '''
        This function should register model and
        write keypoints and descriptors in self.kp and self.des
        :return: None
        '''
        pass


    def upload_image(self, path: str) -> None:
        '''
        This func should upload image using cv
        from given path and save to self.img
        :param path: str
        :return: None
        '''
        pass

    def _crop_image(self, img: np.ndarray, points: np.ndarray) -> None:
        '''
        This function should crop image
        by given points
        :param img: np.ndarray
        :return: None
        '''
        pass



