import cv2 as cv
import numpy as np
from src.registration.registration import Registration
from src.registration.registration_ui import RegistrationUI


class RectangleModel():
    def __init__(self, img: np.ndarray = None, output_path: str = '', feature_method: str = '',
                 key_points_2d: list = None, key_points_3d: list = None, des: np.ndarray = np.empty((0, 0)),
                 object_corners_2d: list = None, object_corners_3d: list = None, horizontal_size: float = 0.0,
                 vertical_size: float = 0.0):
        '''
        Initializes the RectangleModel class with the provided parameters.

        :param img: np.ndarray, grayscale image of the object (default: None)
        :param output_path: str, path to save the processed image
        :param feature_method: str, feature detection method used (e.g., "ORB", "SIFT")
        :param key_points_2d: list, detected key points in 2d
        :param key_points_3d: list, detected key points in 3d
        :param des: np.ndarray, descriptors corresponding to the key points
        :param object_corners_2d: list, 2D object corners for registration
        :param object_corners_3d: list, 3D object corners for registration
        :param horizontal_size: float, horizontal size of the object (width) for Olga
        :param vertical_size: float, vertical size of the object (height)
        '''
        self.img = img
        self.output_path = output_path
        self.feature_method = feature_method
        self.key_points_2d = key_points_2d
        self.key_points_3d = key_points_3d
        self.des = des
        self.object_corners_2d = object_corners_2d
        self.object_corners_3d = object_corners_3d
        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size

    def __str__(self):
        return (
            f"RectangleModel:\n"
            f"  img: {'Loaded' if self.img is not None else 'Not Loaded'}\n"
            f"  output_path: {self.output_path}\n"
            f"  feature_method: {self.feature_method}\n"
            f"  key_points_2d: {self.key_points_2d}\n"
            f"  key_points_3d: {self.key_points_3d}\n"
            f"  des: {self.des}\n"
            f"  object_corners_2d: {self.object_corners_2d}\n"
            f"  object_corners_3d: {self.object_corners_3d}\n"
            f"  horizontal_size: {self.horizontal_size}\n"
            f"  vertical_size: {self.vertical_size}"
        )

    def upload_image(self, input_path: str, output_path: str) -> None:
        '''
        Loads a grayscale image from the specified file path and saves it to the output path.

        :param input_path: str, path to the input image file
        :param output_path: str, path to save the processed image
        :return: None
        '''
        try:
            self.img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            self.output_path = output_path
            cv.imwrite(self.output_path, self.img)
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def calculate_object_size(self):
        '''Calculates horizontal_size and vertical_size of the object.'''
        if self.object_corners_2d.shape != (4, 2):
            raise ValueError("Передайте массив из 4 точек с координатами (x, y).")

        horizontal_size = np.linalg.norm(self.object_corners_2d[1] - self.object_corners_2d[0])
        vertical_size = np.linalg.norm(self.object_corners_2d[3] - self.object_corners_2d[0])
        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size

    def save_to_npz(self, filename: str) -> None:
        '''
        Saves the model's attributes to a .npz file.

        :param filename: str, path to save the .npz file
        :return: None
        '''
        keypoints = [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle, 'response': kp.response,
                      'octave': kp.octave, 'class_id': kp.class_id} for kp in self.key_points_2d]

        np.savez(
            filename,
            output_path=self.output_path or "",
            feature_method=self.feature_method or "",
            key_points_2d=keypoints if keypoints is not None else [],
            key_points_3d=self.key_points_3d if self.key_points_3d is not None else [],
            des=self.des if self.des is not None and self.des.size > 0 else np.empty((0, 0)),
            object_corners_2d=self.object_corners_2d if self.object_corners_2d is not None and len(
                self.object_corners_2d) > 0 else [],
            object_corners_3d=self.object_corners_3d if self.object_corners_3d is not None and len(
                self.object_corners_3d) > 0 else [],
            horizontal_size=self.horizontal_size,
            vertical_size=self.vertical_size
        )

    @classmethod
    def load(cls, filename: str) -> 'RectangleModel':
        '''
        Loads a RectangleModel instance from a .npz file.

        :param filename: str, path to the .npz file
        :return: RectangleModel, an instance of the RectangleModel class
        '''
        data = np.load(filename, allow_pickle=True)
        keys = ["output_path", "feature_method", "key_points_2d", "key_points_3d", "des", "object_corners_2d",
                "object_corners_3d"]

        for key in keys:
            if key not in data:
                raise KeyError(f"Missing key {key} in the loaded data.")

        keypoints = [cv.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'],
                                 kp['response'], kp['octave'], kp['class_id'])
                     for kp in data['key_points_2d']]

        new_object = cls(
            img=None,
            output_path=data["output_path"].item() if isinstance(data["output_path"], np.ndarray) else data[
                "output_path"],
            feature_method=data.get("feature_method"),
            key_points_2d=keypoints,
            key_points_3d=data["key_points_3d"],
            des=data.get("des"),
            object_corners_2d=data.get("object_corners_2d"),
            object_corners_3d=data.get("object_corners_3d"),
            horizontal_size=data.get("horizontal_size"),
            vertical_size=data.get("vertical_size")
        )

        output_path = str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                          else data['output_path'])
        new_object.upload_image(output_path, output_path)
        return new_object

    def register_and_save_rectangular_model(self, img_path: str, output_image: str, feature_method: str,
                                            key_points_2d: list, key_points_3d: list,
                                            des: np.ndarray, object_corners_2d: list, object_corners_3d: list,
                                            model_output: str) -> None:
        '''
        Registers the rectangular model using provided data and saves it to a file.

        :param img_path: str, path to the input image
        :param output_image: str, path to save the processed image
        :param feature_method: str, feature detection method used
        :param key_points_2d: list, detected key points in 2d
        :param key_points_3d: list, detected key points in 3d
        :param des: np.ndarray, descriptors corresponding to the key points
        :param object_corners_2d: list, 2D object corners for registration (default: None)
        :param object_corners_3d: list, 3D object corners for registration (default: None)
        :param model_output: str, path to save the model
        :return: None
        '''

        self.upload_image(img_path, output_image)
        self.feature_method = feature_method
        self.key_points_2d = key_points_2d
        self.key_points_3d = key_points_3d
        self.des = des
        self.object_corners_2d = object_corners_2d
        self.object_corners_3d = object_corners_3d
        self.calculate_object_size()
        self.save_to_npz(model_output)
        print(f"rectangle model saved to {model_output}")


def register(input_image: str, output_image: str, object_corners_3d: np.ndarray,
             crop_method: str, feature_method: str, model_output: str) -> None:
    '''
    Main registration function to detect, register, and save a rectangular model.

    :param input_image: str, path to the input image
    :param output_image: str, path to save the processed image
    :param object_corners_3d: np.ndarray, 3D coordinates of the object's corners
    :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
    :param feature_method: str, feature detection method to use (e.g., "ORB", "SIFT")
    :param model_output: str, path to save the model as a .npz file
    :return: None
    '''
    corners_model = RegistrationUI()
    object_corners_2d, object_corners_3d = corners_model.register_object_corners(input_image, object_corners_3d,
                                                                                 crop_method)

    registration_model = Registration()
    key_points_2d, key_points_3d, des = registration_model.register_with_object_corners(input_image, feature_method,
                                                                                        object_corners_2d,
                                                                                        object_corners_3d)

    rect_model = RectangleModel()
    rect_model.register_and_save_rectangular_model(input_image, output_image, feature_method, key_points_2d,
                                                   key_points_3d, des, object_corners_2d, object_corners_3d,
                                                   model_output)
