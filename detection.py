import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from registration import Model

class Detector():


    def __init__(self):
        self.model: Model = Model()
        self.images: np.ndarray((-1, 2))


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        :return: None
        '''
        pass


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