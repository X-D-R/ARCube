import cv2 as cv
import numpy as np


class RegistrationUI():
    def __init__(self, img: np.ndarray = None, object_corners_2d: list = None, object_corners_3d: list = None):
        '''
        Initializes the RegistrationUI class with optional parameters.

        :param img: np.ndarray, grayscale image of the object (default: None)
        :param object_corners_2d: list, 2D object corners for registration (default: None)
        :param object_corners_3d: list, 3D object corners for registration (default: None)
        '''
        self.img = img
        self.object_corners_2d = object_corners_2d
        self.object_corners_3d = object_corners_3d

    def upload_image(self, input_path: str) -> None:
        '''
        Loads a grayscale image from the specified file path.

        :param input_path: str, path to the input image file
        :return: None
        '''
        try:
            self.img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def insert_object_corners_3d(self, object_corners_3d: np.ndarray) -> None:
        '''
        Sets the 3D object corners.

        :param object_corners_3d: np.ndarray, 3D coordinates of the object corners
        :return: None
        '''
        self.object_corners_3d = object_corners_3d

    def select_object_corners(self, crop_method: str) -> None:
        '''
        Allows the user to select 2D object corners interactively or automatically.

        :param crop_method: str, the method for selecting corners ("photo" or "corner")
        :return: None
        '''
        if self.img is None:
            raise ValueError("Image must be loaded before selecting corners.")

        points_2d = []
        h, w = self.img.shape
        if crop_method == 'photo':
            max_height = 800
            scale = 1.0
            image = self.img
            if h > max_height:
                scale = max_height / h
                image = cv.resize(image, (int(w * scale), int(h * scale)))

            def click_event(event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and len(points_2d) < 7:
                    points_2d.append([x, y])
                    cv.circle(image, (x, y), 5, (0, 255, 0), -1)
                    cv.imshow("Select Corners", image)
                    print(f"Selected corner: ({x}, {y})")

            print("Mark object corners (from 4 to 7 points):")
            cv.imshow("Select Corners", image)
            cv.setMouseCallback("Select Corners", click_event)
            cv.waitKey(0)
            cv.destroyAllWindows()

            if len(points_2d) < 4:
                raise ValueError("At least 4 points are required to define the object.")
        elif crop_method == 'corner':
            points_2d.extend([[0, 0], [w, 0], [w, h], [0, h]])
        else:
            raise ValueError("You chose wrong crop method, use 'photo' or 'corner' ")
        self.object_corners_2d = np.array(points_2d, dtype="float32")
        print(f"Selected corners: {self.object_corners_2d}")

    def get_2d_3d_corners(self) -> list:
        '''
        Combines 2D and 3D corner pairs into a single list.

        :return: list of dictionaries where each dictionary contains a 2D-3D pair
        '''
        assert len(self.object_corners_2d) == len(self.object_corners_3d), \
            "Number of 2D and 3D corners must match."

        corners_2d_3d = [
            {'2d': self.object_corners_2d[i], '3d': self.object_corners_3d[i]}
            for i in range(len(self.object_corners_2d))
        ]
        return corners_2d_3d

    def register_object_corners(self, img_path: str, object_corners_3d: np.ndarray, crop_method: str) -> list:
        '''
        Performs the full process of registering object corners, including loading the image,
        setting 3D corners, and selecting 2D corners interactively or automatically.

        :param img_path: str, path to the input image
        :param object_corners_3d: np.ndarray, 3D coordinates of the object corners
        :param crop_method: str, method for selecting 2D corners ("photo" or "corner")
        :return: list of dictionaries, each containing a 2D-3D corner pair
        '''
        self.upload_image(img_path)
        self.insert_object_corners_3d(object_corners_3d)
        self.select_object_corners(crop_method)
        corners_2d_3d = self.get_2d_3d_corners()
        print("Registration object corners completed!")
        return corners_2d_3d
