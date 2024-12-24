import cv2 as cv
import numpy as np
from Draw_functions import upload_image


class RectangleModelDrawer:
    def __init__(self):
        pass

    @staticmethod
    def draw_contours_of_box(image_path: str, output_path: str, img_points: np.ndarray, color=(0, 0, 255),
                             thickness=15) -> None:
        '''
        This func draw contours of box
        and save image to selected path
        :param image_path: str, path to source image
        :param output_path: str, path to output image
        :param img_points: np.ndarray, numpy array of 2D points on image, that is 8 points of original box
        :param color: tuple, RGB tuple, for example RED is (255, 0, 0)
        :param thickness: int, the thickness of lines
        '''
        img = upload_image(image_path)
        img = cv.polylines(img, [np.int32(img_points[::2])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[1::2])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[:2:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[2:4:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[4:6:])], True, color, thickness)
        img = cv.polylines(img, [np.int32(img_points[6:8:])], True, color, thickness)
        cv.imwrite(output_path, img)
        return
