import cv2 as cv
from src.registration.rectangle_model import RectangleModel
from src.detection.detection import Detector


class RectangleModelDrawerDebug:
    def __init__(self):
        pass

    @staticmethod
    def draw_matches(model: RectangleModel, detector: Detector, image_path: str, output_path: str) -> None:
        '''
        This func draw matches
        and save image to selected path
        :param model: Model, model of object
        :param detector: Detector, detector object
        :param image_path: str, path to source image
        :param output_path: str, path to where save image
        '''
        img = cv.imread(image_path)
        kp2, des2 = detector.descriptor.detectAndCompute(img, None)
        img_points, inliers_original, inliers_frame, good, mask = detector.detect(image_path)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask,  # draw only inliers
                           flags=2)
        img_res = cv.drawMatches(model.img, model.kp, img, kp2, good, None, **draw_params)
        cv.imwrite(output_path, img_res)
        return
