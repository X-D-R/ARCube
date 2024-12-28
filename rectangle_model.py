import cv2 as cv
import numpy as np
from registration import Registration
from registation_ui import RegistrationUI


class RectangleModel():
    def __init__(self, img: np.ndarray = None, output_path: str = '', feature_method: str = '',
                 key_points_2d: list = None, key_points_3d: list = None, des: np.ndarray = np.empty((0, 0)),
                 object_corners_2d: list = None, object_corners_3d: list = None):
        '''
        Initializes the RectangleModel class with the provided parameters.

        :param img: np.ndarray, grayscale image of the object (default: None)
        :param output_path: str, path to save the processed image
        :param feature_method: str, feature detection method used (e.g., "ORB", "SIFT")
        :param key_points_2d: list, detected key points in 2d
        :param key_points_3d: list, detected key points in 3d
        :param des: np.ndarray, descriptors corresponding to the key points
        :param object_corners_2d: list, 2D object corners for registration (default: None)
        :param object_corners_3d: list, 3D object corners for registration (default: None)
        '''
        self.img = img
        self.output_path = output_path
        self.feature_method = feature_method
        self.key_points_2d = key_points_2d
        self.key_points_3d = key_points_3d
        self.des = des
        self.object_corners_2d = object_corners_2d
        self.object_corners_3d = object_corners_3d

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
        np.savez(
            filename,
            output_path=self.output_path or "",
            feature_method=self.feature_method or "",
            key_points_2d=keypoints_to_array(self.key_points_2d) if self.key_points_2d is not None else [],
            key_points_3d=keypoints_to_array(self.key_points_3d) if self.key_points_3d is not None else [],
            des=self.des if self.des is not None and self.des.size > 0 else np.empty((0, 0)),
            object_corners_2d=self.object_corners_2d if self.object_corners_2d is not None and len(
                self.object_corners_2d) > 0 else [],
            object_corners_3d=self.object_corners_3d if self.object_corners_3d is not None and len(
                self.object_corners_3d) > 0 else []
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

        new_object = cls(
            img=None,
            output_path=data["output_path"].item() if isinstance(data["output_path"], np.ndarray) else data[
                "output_path"],
            feature_method=data.get("feature_method"),
            key_points_2d=array_to_keypoints(data["key_points_2d"]) if "key_points_2d" in data else [],
            key_points_3d=array_to_keypoints(data["key_points_3d"]) if "key_points_3d" in data else [],
            des=data.get("des"),
            object_corners_2d=data.get("object_corners_2d"),
            object_corners_3d=data.get("object_corners_3d"),
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
        self.save_to_npz(model_output)
        print(f"rectangle model saved to {model_output}")


def keypoints_to_array(keypoints):
    """Convert cv2.KeyPoint objects to a serializable format."""
    if len(keypoints) > 0 and isinstance(keypoints[0], cv.KeyPoint):
        return np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in keypoints])
    return np.array(keypoints)  # Assume already serialized if not cv2.KeyPoint



def array_to_keypoints(array):
    '''Convert an array back to cv2.KeyPoint objects.'''
    return [cv.KeyPoint(x=pt[0], y=pt[1], size=pt[2], angle=pt[3],
                        response=pt[4], octave=int(pt[5]), class_id=int(pt[6])) for pt in array]


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


# example
object_corners_3d = np.array([
    [0, 0, 0],  # Top-left
    [13, 0, 0],  # Top-right
    [13, 20.5, 0],  # Bottom-right
    [0, 20.5, 0],  # Bottom-left
    # Optionally, add more points if needed
], dtype="float32")

register(
    input_image="old_files/andrew photo video/reference messy.jpg",
    output_image="output_script_test.jpg",
    object_corners_3d=object_corners_3d,
    crop_method='corner',
    feature_method="ORB",
    model_output="model_script_test.npz"
)

model = RectangleModel.load("model_script_test.npz")
print(model.img)
print(model.output_path)
print(model.feature_method)
print(model.key_points_2d)
print(model.key_points_3d)
print(model.des)
print(model.object_corners_2d)
print(model.object_corners_3d)
