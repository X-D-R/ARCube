import cv2 as cv
import numpy as np


class Registration():
    '''
    Registration class to perform image registration.
    '''

    def __init__(self, img: np.ndarray = None,  kp: list = None, des: np.ndarray = np.empty((0, 0)), method: str = '',
                 points_2d_3d: list = None, object_corners_2d: list = None, object_corners_3d: list = None):
        '''
        Initializes the Model class with the provided parameters.

        :param img: np.ndarray, the image data
        :param kp: list, keypoints detected in the image
        :param des: np.ndarray, descriptors for the keypoints
        :param method: str, feature detection method used
        :param points_2d_3d: list, list of 2D-3D key point pairs
        :param object_corners_2d: list, 2D object corners for registration
        :param object_corners_3d: list, 3D object corners for registration
        '''
        self.img = img
        self.kp = kp
        self.des = des
        self.method = method
        self.points_2d_3d = points_2d_3d if points_2d_3d is not None else []
        self.object_corners_2d = object_corners_2d
        self.object_corners_3d = object_corners_3d


    def upload_image(self, input_path: str) -> None:
        '''
        Loads the image of the object.

        :param input_path: str, path to the image file
        :return: None
        '''
        try:
            self.img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def register(self, feature: str) -> None:
        '''
        Detects keypoints and computes descriptors using the specified feature detection method.
        Filters keypoints and descriptors to include only those inside the defined object area.

        :param feature: str, the feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
        :return: None
        '''
        assert self.img is not None, 'Image should be loaded first'
        assert self.object_corners_2d is not None, 'Object corners must be defined'

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

        keypoints, descriptors = method.detectAndCompute(self.img, None)

        object_polygon = np.array(self.object_corners_2d, dtype="float32")
        filtered_kp = []
        filtered_des = []

        for kp, des in zip(keypoints, descriptors):
            point = np.array(kp.pt, dtype="float32")
            if cv.pointPolygonTest(object_polygon, point, False) >= 0:
                filtered_kp.append(kp)
                filtered_des.append(des)

        self.kp = filtered_kp
        self.des = np.array(filtered_des) if filtered_des else None
        self.method = feature

    def split_2d_3d_corners(self, corners_2d_3d: list) -> None:
        '''
        Splits a combined 2D-3D corners list into separate 2D and 3D arrays.

        :param corners_2d_3d: list of dictionaries where each dictionary contains a 2D-3D pair
        :return: None
        '''
        self.object_corners_2d = np.array([corner['2d'] for corner in corners_2d_3d], dtype="float32")
        self.object_corners_3d = np.array([corner['3d'] for corner in corners_2d_3d], dtype="float32")

    def register_point(self, point_2d: list, point_3d: list) -> None:
        '''
        Registers a single 2D-3D point pair.

        :param point_2d: list, 2D coordinates of the point
        :param point_3d: list, 3D coordinates of the point
        :return: None
        '''
        self.points_2d_3d.append({'2d': point_2d, '3d': point_3d})
        print(f"Registered 2D point {point_2d} with 3D point {point_3d}")

    def compute_plane(self) -> tuple:
        '''
        Computes the plane equation Ax + By + Cz + D = 0 using 3D points.

        :return: tuple, coefficients (A, B, C, D) of the plane equation
        '''
        points_3d = self.object_corners_3d
        p1, p2, p3 = points_3d[0], points_3d[1], points_3d[-1]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        A, B, C = normal
        D = -np.dot(normal, p1)
        return A, B, C, D

    def map_keypoints_to_3d_plane(self) -> None:
        '''
        Maps 2D keypoints to 3D coordinates using the plane equation.

        :return: None
        '''
        assert self.object_corners_2d is not None, "2D corners must be defined."
        assert len(self.object_corners_2d) == len(self.object_corners_3d), \
            "Number of 2D and 3D corners must match."

        bounds_2d = self.object_corners_2d.max(axis=0) - self.object_corners_2d.min(axis=0)
        bounds_3d = self.object_corners_3d.max(axis=0) - self.object_corners_3d.min(axis=0)
        scale_x = bounds_3d[0] / bounds_2d[0]
        scale_y = bounds_3d[1] / bounds_2d[1]

        A, B, C, D = self.compute_plane()

        for kp in self.kp:
            point_2d = np.array([kp.pt[0], kp.pt[1]])

            x = (point_2d[0] - self.object_corners_2d.min(axis=0)[0]) * scale_x
            y = (point_2d[1] - self.object_corners_2d.min(axis=0)[1]) * scale_y
            z = -(A * x + B * y + D) / C

            self.register_point([point_2d[0], point_2d[1]], [x, y, z])

    def register_with_object_corners(self, feature_method: str, corners_2d_3d: list) -> None:
        '''
        Performs registration, including interactive corner selection.

        :param feature_method: str, the feature detection method to use (e.g., "ORB", "SIFT")
        :param object_corners_3d: np.ndarray, the 3D coordinates of the object corners
        :return: None
        '''
        self.split_2d_3d_corners(corners_2d_3d)
        self.register(feature_method)
        self.map_keypoints_to_3d_plane()
        print("Registration completed with object corners.")


def register(input_image: str, corners_2d_3d: list, feature_method: str, model_output: str) -> None:

    '''
    Main registration function.

    :param camera_params: str, path to the camera parameters .npz file
    :param input_image: str, path to the input image
    :param output_image: str, path to save the output image
    :param vol: int, volume of the object
    :param object_corners_3d: np.ndarray, the 3D coordinates of the object corners
    :param crop_method: str, the method used for cropping ("photo" or "corner")
    :param feature_method: str, the feature detection method to use (e.g., "ORB")
    :param model_output: str, path to save the model output .npz file
    :return: None
    '''
    model = Registration()
    model.upload_image(input_image)
    model.register_with_object_corners(feature_method, corners_2d_3d)
    model.save_to_npz(model_output)
    print(f"Model saved to {model_output}")


# example
object_corners_3d = np.array([
    [0, 0, 0],  # Top-left
    [13, 0, 0],  # Top-right
    [13, 20.5, 0],  # Bottom-right
    [0, 20.5, 0],  # Bottom-left
    # Optionally, add more points if needed
], dtype="float32")

register(
    camera_params="CameraParams/cam_params_andrew.npz",
    input_image="old_files/andrew photo video/reference messy.jpg",
    output_image="output_script_test.jpg",
    vol=0,
    object_corners_3d=object_corners_3d,
    crop_method='corner',
    feature_method="ORB",
    model_output="model_script_test.npz"
)

