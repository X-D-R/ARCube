import cv2 as cv
import numpy as np
from registration import Registration
from registation_ui import RegistrationUI


class RectangleModel():
    def __init__(self, img: np.ndarray = None, output_path: str = '', feature_method: str = '',
                 points_2d_3d_des: list = None, corners_2d_3d: list = None):
        '''
        Initializes the RectangleModel class with the provided parameters.

        :param img: np.ndarray, grayscale image of the object (default: None)
        :param output_path: str, path to save the processed image
        :param feature_method: str, feature detection method used (e.g., "ORB", "SIFT")
        :param points_2d_3d_des: list, list of 2D-3D keypoint pairs with descriptors
        :param corners_2d_3d: list, list of 2D-3D object corner pairs
        '''
        self.img = img
        self.output_path = output_path
        self.feature_method = feature_method
        self.points_2d_3d_des = points_2d_3d_des
        self.corners_2d_3d = corners_2d_3d

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

    def save_to_npz(self, filename: str) -> None:
        '''
        Saves the model's attributes to a .npz file.

        :param filename: str, path to save the .npz file
        :return: None
        '''
        np.savez(filename, output_path=self.output_path, feature_method=self.feature_method,
                 points_2d_3d_des=self.points_2d_3d_des, corners_2d_3d=self.corners_2d_3d)

    @classmethod
    def load(cls, filename: str) -> 'RectangleModel':
        '''
        Loads a RectangleModel instance from a .npz file.

        :param filename: str, path to the .npz file
        :return: RectangleModel, an instance of the RectangleModel class
        '''
        data = np.load(filename, allow_pickle=True)

        if 'output_path' not in data:
            raise ValueError("The file does not contain the 'output_path' attribute.")

        new_object = cls(
            img=None,
            output_path=str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                            else data['output_path']),
            feature_method=data['feature_method'] if 'feature_method' in data else None,
            points_2d_3d_des=data['points_2d_3d_des'],
            corners_2d_3d=data['corners_2d_3d'],
        )
        output_path = str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                          else data['output_path'])
        new_object.upload_image(output_path, output_path)
        return new_object

    def register_and_save_rectangular_model(self, img_path: str, output_image: str, feature_method: str,
                                            points_2d_3d_des: list, corners_2d_3d: list, model_output: str) -> None:
        '''
        Registers the rectangular model using provided data and saves it to a file.

        :param img_path: str, path to the input image
        :param output_image: str, path to save the processed image
        :param feature_method: str, feature detection method used
        :param points_2d_3d_des: list, list of 2D-3D keypoint pairs with descriptors
        :param corners_2d_3d: list, list of 2D-3D corner pairs
        :param model_output: str, path to save the model
        :return: None
        '''
        if not points_2d_3d_des or not corners_2d_3d:
            raise ValueError("Points and corners data must be provided for registration.")

        self.upload_image(img_path, output_image)
        self.method = feature_method
        self.points_2d_3d_des = points_2d_3d_des
        self.corners_2d_3d = corners_2d_3d
        self.save_to_npz(model_output)
        print(f"rectangle model saved to {model_output}")


def register(input_image: str, output_image: str, object_corners_3d: np.ndarray,
             crop_method: str, feature_method: str, model_output: str) -> None:
    '''
    Main registration function to detect, register, and save a rectangular model.

    :param input_image: str, path to the input image
    :param output_image: str, path to save the processed image
    :param object_corners_3d: np.ndarray, 3D coordinates of the object's corners
    :param crop_method: str, method for selecting 2D corners ("photo" or "corner")
    :param feature_method: str, feature detection method to use (e.g., "ORB", "SIFT")
    :param model_output: str, path to save the model as a .npz file
    :return: None
    '''
    corners_model = RegistrationUI()
    corners_2d_3d = corners_model.register_object_corners(input_image, object_corners_3d, crop_method)

    registration_model = Registration()
    points_2d_3d_des = registration_model.register_with_object_corners(input_image, feature_method, corners_2d_3d)

    rect_model = RectangleModel()
    rect_model.register_and_save_rectangular_model(input_image, output_image, feature_method, points_2d_3d_des,
                                                   corners_2d_3d, model_output)


# # example
# object_corners_3d = np.array([
#     [0, 0, 0],  # Top-left
#     [13, 0, 0],  # Top-right
#     [13, 20.5, 0],  # Bottom-right
#     [0, 20.5, 0],  # Bottom-left
#     # Optionally, add more points if needed
# ], dtype="float32")
#
# register(
#     input_image="old_files/andrew photo video/reference messy.jpg",
#     output_image="output_script_test.jpg",
#     object_corners_3d=object_corners_3d,
#     crop_method='corner',
#     feature_method="ORB",
#     model_output="model_script_test.npz"
# )
#
