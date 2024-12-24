import cv2 as cv
import numpy as np
from Draw_functions import upload_image
from registration import Model
from detection import Detector


class RectangleModelDrawerDebug:
    def __init__(self):
        pass

    @staticmethod
    def draw_matches(model: Model, detector: Detector, image_path: str, output_path: str) -> None:
        '''
        This func draw matches
        and save image to selected path
        :param model: Model, model of object
        :param detector: Detector, detector object
        :param image_path: str, path to source image
        :param output_path: str, path to where save image
        '''
        img = upload_image(image_path)
        kp2, des2 = detector.descriptor.detectAndCompute(img, None)
        img_points, inliers_original, inliers_frame, good, mask = detector.detect(image_path)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask,  # draw only inliers
                           flags=2)
        img_res = cv.drawMatches(model.img, model.kp, img, kp2, good, None, **draw_params)
        cv.imwrite(output_path, img_res)
        return
