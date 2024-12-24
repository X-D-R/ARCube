import cv2 as cv
import numpy as np

class RegistrationUI():

    def __init__(self, img: np.ndarray = None, height: int = 0, width: int = 0, object_corners_2d: list = None,
                 object_corners_3d: list = None):
        '''
        Initializes the RegistrationUI class with the provided parameters.

        :param img: np.ndarray, the image data
        :param height: int, height of the image
        :param width: int, width of the image
        :param object_corners_2d: list, 2D object corners for registration
        :param object_corners_3d: list, 3D object corners for registration
        '''
        self.img = img
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

    def select_object_corners(self, crop_method: str) -> None:
        '''
        Allows the user to select the object corners interactively based on the crop method.

        :param crop_method: str, the method for selecting corners ("photo" or "corner")
        :return: None
        '''
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

            instructions = f"Mark object corners (from 4 to 7 points):"
            print(instructions)
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


