import cv2 as cv
import numpy as np
import os.path

MAIN_DIR = os.path.split(os.path.abspath("main.py"))[0]


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
            image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            '''image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            image = cv.filter2D(image, -1, kernel)
            image = cv.medianBlur(image, 5)
            image = cv.equalizeHist(image)
            _, image = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            image = cv.erode(image, kernel, iterations=1)
            image = cv.dilate(image, kernel, iterations=1)
            adaptive_thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                    11, 2)
            image = cv.GaussianBlur(image, (11, 11), 2)
            adaptive_thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                   11, 2)'''
            self.img = image
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def insert_object_corners_3d(self, object_corners_3d: np.ndarray) -> None:
        '''
        Sets the 3D object corners.

        :param object_corners_3d: np.ndarray, 3D coordinates of the object corners
        :return: None
        '''
        self.object_corners_3d = object_corners_3d

    #def select_object_corners(self, crop_method: str) -> None:
    def select_object_corners(self) -> None:
        '''
        Allows the user to select 2D object corners interactively or automatically.

        :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
        :return: None
        '''
        if self.img is None:
            raise ValueError("Image must be loaded before selecting corners.")

        points_2d = []
        h, w = self.img.shape
        #if crop_method == 'manual':
        example_image = cv.imread(
            os.path.join(MAIN_DIR, "ExampleFiles", "examples", "images", "corners_choice_example.jpg"), cv.IMREAD_COLOR)

        max_height = 600
        scale1 = 1.0
        h1, w1, channels = example_image.shape

        if h1 > max_height:
            scale1 = max_height / h1
            example_image = cv.resize(example_image, (int(w1 * scale1), int(h1 * scale1)))

        cv.imshow("Selection corners example", example_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        max_height = 600
        scale = 1.0
        image = self.img
        if h > max_height:
            scale = max_height / h
            image = cv.resize(image, (int(w * scale), int(h * scale)))

        def click_event(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN and len(points_2d) < 4:
                points_2d.append([x/scale, y/scale])
                cv.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv.imshow("Select Corners", image)
                print(f"Selected corner: ({x/scale}, {y/scale})")

        print("Mark object corners 4 points:")
        cv.imshow("Select Corners", image)
        cv.setMouseCallback("Select Corners", click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()
        #points_2d = [[0, 0], [w, 0], [w, h], [0, h]]

        if len(points_2d) < 4:
            raise ValueError("At least 4 points are required to define the object.")
        if len(points_2d) != len(self.object_corners_3d):
            raise ValueError("Length of 2d and 3d points must be equal! 4 points are required.")
        '''elif crop_method == 'corner':
            points_2d.extend([[0, 0], [w, 0], [w, h], [0, h]])
        else:
            raise ValueError("You chose wrong crop method, use 'manual' or 'corner' ")'''
        self.object_corners_2d = np.array(points_2d, dtype="float32")
        print(f"Selected corners: {self.object_corners_2d}")

    def register_object_corners(self, img_path: str, object_corners_3d: np.ndarray) -> tuple:
        '''
        Performs the full process of registering object corners, including loading the image,
        setting 3D corners, and selecting 2D corners interactively or automatically.

        :param img_path: str, path to the input image
        :param object_corners_3d: np.ndarray, 3D coordinates of the object corners
        :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
        :return: tuple of 2 lists with  2D, 3D corners
        '''
        self.upload_image(img_path)
        self.insert_object_corners_3d(object_corners_3d)
        self.select_object_corners()
        object_corners_2d = self.object_corners_2d
        object_corners_3d = self.object_corners_3d
        print("Registration object corners completed!")
        return object_corners_2d, object_corners_3d
