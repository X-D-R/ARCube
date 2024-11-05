import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Model():

    def __init__(self):
        '''
        Attributes:
        - img (np.ndarray): The image data
        - kp (tuple): Key points detected in the image
        - des (np.ndarray): Descriptors for the key points
        - height (int): The height of the object (needed for 3d rectangle frame)
        - camera_params (dict): Dictionary containing camera parameters
        '''
        self.img: np.ndarray = np.empty((0, 0))
        self.kp: list = []
        self.des: np.ndarray = np.empty((0, 0))
        self.height: int = 0
        self.camera_params: dict = {}


    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        path should be .npz file
        :return: None
        '''
        with np.load(path) as file:
            self.camera_params = {
                "mtx": file['cameraMatrix'],
                "dist": file['dist'],
                "rvecs": file['rvecs'],
                "tvecs": file['tvecs']
            }


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
        :param feature: Feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
        :return: None
        '''
        if feature == "ORB":
            method = cv.ORB.create()
        elif feature == "KAZE":
            method = cv.KAZE.create()
        elif feature == "AKAZE":
            method = cv.AKAZE.create()
        elif feature == "BRISK":
            method = cv.BRISK.create()
        elif feature == "SIFT":
            method = cv.SIFT.create()
        else:
            raise ValueError("Unsupported feature type.")

        self.kp, self.des = method.detectAndCompute(self.img, None)

    def crop_image_by_points(self, points: np.ndarray) -> None:
        '''
        This function should crop image
        by given points
        :param points: np.ndarray of shape (4, 2)
        :return: None
        '''
        if len(points) == 4:
            rect = np.array(points, dtype="float32")

            width_a = np.linalg.norm(rect[0] - rect[1])
            width_b = np.linalg.norm(rect[2] - rect[3])
            max_width = int(max(width_a, width_b))

            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[3] - rect[0])
            max_height = int(max(height_a, height_b))

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")

            matrix = cv.getPerspectiveTransform(rect, dst)
            warped = cv.warpPerspective(self.img, matrix, (max_width, max_height))
            self.img = warped

        else:
            print("Error: Insufficient points selected!")


    def click_event(self, event, x, y, param):
        '''
        This function captures four points on the image upon mouse clicks.
        :param event: Mouse event, like left button click
        :param x: X-coordinate of the click
        :param y: Y-coordinate of the click
        :param param: Tuple containing the display image and the points list(np.ndarray)
        :return: None
        '''
        image, points = param
        if event == cv.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Select Points', image)
            if len(points) == 4:
                cv.destroyWindow('Select Points')


    def crop_image_by_clicks(self) -> None:
        '''
        This function should crop image
        by points that you choose on picture
        :return: None
        '''
        points = []
        image=self.img.copy()
        h, w = image.shape[:2]
        max_height = 800
        if h > max_height:
            scale = max_height / h
            image = cv.resize(image, (int(w * scale), int(h * scale)))

        display_image = image.copy()
        cv.imshow('Select Points', display_image)
        cv.setMouseCallback('Select Points', self.click_event, param=(display_image, points))
        cv.waitKey(0)

        if len(points) == 4:
            points = np.array(points, dtype="float32")
            self.crop_image_by_points(points)
        else:
            print("Error: Insufficient points selected!")



