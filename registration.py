import cv2 as cv
import numpy as np


class Registration():
    def __init__(self, img: np.ndarray = None, kp: list = None, des: np.ndarray = np.empty((0, 0)), method: str = '',
                 points_2d_3d_des: list = None, object_corners_2d: list = None, object_corners_3d: list = None):
        '''
        Initializes the Registration class with the provided parameters.

        :param img: np.ndarray, grayscale image of the object
        :param kp: list, detected keypoints
        :param des: np.ndarray, descriptors corresponding to the keypoints
        :param method: str, feature detection method used (e.g., "ORB", "SIFT")
        :param points_2d_3d_des: list, list of dictionaries containing 2D-3D keypoint pairs and their descriptors
        :param object_corners_2d: list, 2D coordinates of the object's corners
        :param object_corners_3d: list, 3D coordinates of the object's corners
        '''
        self.img = img
        self.kp = kp
        self.des = des
        self.method = method
        self.points_2d_3d_des = points_2d_3d_des if points_2d_3d_des is not None else []
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
            if self.img is None:
                raise ValueError("Failed to load the image. Check the file path.")
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def find_kp(self, feature: str) -> None:
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
        Splits a combined list of 2D-3D corner pairs into separate 2D and 3D corner arrays.

        :param corners_2d_3d: list, list of dictionaries containing 2D and 3D corners
        :return: None
        '''
        self.object_corners_2d = np.array([corner['2d'] for corner in corners_2d_3d], dtype="float32")
        self.object_corners_3d = np.array([corner['3d'] for corner in corners_2d_3d], dtype="float32")

    def register_point(self, point_2d: list, point_3d: list, des) -> None:
        '''
        Registers a single 2D-3D point pair.

        :param point_2d: list, 2D coordinates of the point
        :param point_3d: list, 3D coordinates of the point
        :param des: description of the point
        :return: None
        '''
        self.points_2d_3d_des.append({'2d': point_2d, '3d': point_3d, 'des': des})
        print(f"Registered 2D point {point_2d} with 3D point {point_3d} with description {des}")

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
        if self.des is None or len(self.kp) != len(self.des):
            raise ValueError("Keypoints and descriptors must match.")
        assert self.object_corners_2d is not None, "2D corners must be defined."
        assert len(self.object_corners_2d) == len(self.object_corners_3d), \
            "Number of 2D and 3D corners must match."

        bounds_2d = self.object_corners_2d.max(axis=0) - self.object_corners_2d.min(axis=0)
        bounds_3d = self.object_corners_3d.max(axis=0) - self.object_corners_3d.min(axis=0)
        scale_x = bounds_3d[0] / bounds_2d[0]
        scale_y = bounds_3d[1] / bounds_2d[1]

        A, B, C, D = self.compute_plane()

        i = 0
        for kp in self.kp:
            point_2d = np.array([kp.pt[0], kp.pt[1]])

            x = (point_2d[0] - self.object_corners_2d.min(axis=0)[0]) * scale_x
            y = (point_2d[1] - self.object_corners_2d.min(axis=0)[1]) * scale_y
            z = -(A * x + B * y + D) / C

            self.register_point([point_2d[0], point_2d[1]], [x, y, z], self.des[i])
            i += 1

    def get_2d_3d_kp_des(self):
        return self.points_2d_3d_des

    def register_with_object_corners(self, img_path: str, feature_method: str, corners_2d_3d: list) -> list:
        '''
        Performs registration, including interactive corner selection.

        :param img_path: str, path to the image
        :param feature_method: str, the feature detection method to use (e.g., "ORB", "SIFT")
        :param corners_2d_3d: np.ndarray, the 3D coordinates of the object corners
        :return: None
        '''
        self.upload_image(img_path)
        self.split_2d_3d_corners(corners_2d_3d)
        self.find_kp(feature_method)
        self.map_keypoints_to_3d_plane()
        print("Registration object key points completed")
        points_2d_3d_des = self.get_2d_3d_kp_des()
        return points_2d_3d_des
