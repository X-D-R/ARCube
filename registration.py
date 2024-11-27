import numpy as np
import cv2 as cv


class Model():

    def __init__(self, img: np.ndarray = None, output_path: str = '', height: int = 0, width: int = 0, kp: list = None,
                 des: np.ndarray = np.empty((0, 0)), vol: int = 0, camera_params: dict = None, method: str = ''):
        '''
        Attributes:
        - img (np.ndarray): The image data
        - output_path (str): The place where the Model img contains
        - height (int): The height of the picture
        - width (int): The width of the picture
        - kp (list): Key points detected in the image
        - des (np.ndarray): Descriptors for the key points
        - vol (int): The volume of the object (needed for 3d rectangle frame)
        - camera_params (dict): Dictionary containing camera parameters
        - method (str): feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
        '''
        self.img: img
        self.output_path = output_path
        self.height = height
        self.width = width
        self.kp = kp
        self.des = des
        self.vol = vol
        self.camera_params = camera_params
        self.method = method

    def load_camera_params(self, path) -> None:
        '''
        This function should load params from
        file to self.camera_params
        path should be .npz file
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.camera_params = {
                    "mtx": file['cameraMatrix'],
                    "dist": file['dist'],
                    "rvecs": file['rvecs'],
                    "tvecs": file['tvecs']
                }
        else:
            print('Error: it is not .npz file')

    def upload_image(self, input_path: str, output_path: str, vol: int = 0) -> None:
        '''
        This func should upload image using cv
        from given path and save to self.img
        :param input_path: str, to upload photo
        :param output_path: str, to place where the Model img contains
        :param vol: int, the volume of the object (needed for 3d rectangle frame)
        :return: None
        '''
        try:
            self.img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            self.height, self.width = self.img.shape
            self.vol = vol
            self.output_path = output_path
            cv.imwrite(self.output_path, self.img)
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")

    def register(self, feature: str) -> None:
        '''
        This function should register model and
        write keypoints and descriptors in self.kp and self.des
        :param feature: Feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
        :return: None
        '''
        assert self.img is not None, 'Image should be loaded first'
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

        self.kp, self.des = method.detectAndCompute(self.img, None)
        self.method = feature

    def get_params(self) -> (list, np.ndarray):
        return self.kp, self.des

    def order_points(self, pts):
        """ Orders points in the format: top-left, top-right, bottom-right, bottom-left """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def crop_image_by_points(self, points: np.ndarray) -> None:
        '''
        This function should crop image
        by given points
        :param points: np.ndarray of shape (4, 2)
        :return: None
        '''
        if len(points) == 4:
            rect = self.order_points(points)

            width_a = np.linalg.norm(rect[0] - rect[1])
            width_b = np.linalg.norm(rect[2] - rect[3])
            max_width = int(max(width_a, width_b))

            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[3] - rect[0])
            max_height = int(max(height_a, height_b))

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")

            matrix = cv.getPerspectiveTransform(rect, dst)
            warped = cv.warpPerspective(self.img, matrix, (max_width, max_height))
            self.img = warped
            self.height, self.width = self.img.shape
            cv.imwrite(self.output_path, self.img)
        else:
            print("Error: Insufficient points selected!")

    def click_event(self, event, x, y, flags, param):
        '''
        This function captures four points on the image upon mouse clicks.
        :param event: Mouse event, like left button click
        :param x: X-coordinate of the click
        :param y: Y-coordinate of the click
        :param param: Tuple containing the display image and the points list(np.ndarray)
        :return: None
        '''
        image, points = param
        if event == cv.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Select Points', image)

            if len(points) == 4:
                cv.destroyWindow('Select Points')

    def crop_image_by_clicks(self) -> None:
        '''
        This function should crop image
        by points that you choose on picture
        :return: None
        '''
        points = []
        image = self.img.copy()
        h, w = image.shape[:2]
        max_height = 800
        scale = 1.0
        if h > max_height:
            scale = max_height / h
            image = cv.resize(image, (int(w * scale), int(h * scale)))

        display_image = image.copy()
        cv.imshow('Select Points', display_image)
        cv.setMouseCallback('Select Points', self.click_event, param=(display_image, points))
        cv.waitKey(0)

        if len(points) == 4:
            points = np.array([[int(x / scale), int(y / scale)] for [x, y] in points], dtype="float32")
            self.crop_image_by_points(points)
        else:
            print("Error: Insufficient points selected!")

    def _check(self, path_params: str, path_img: str, output_path: str) -> None:
        self.load_camera_params(path_params)
        self.upload_image(path_img, output_path)
        for feature in ["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"]:
            self.register(feature)
            print(f"Feature: {feature}\n\n")
            print(f" KeyPoints: \n {self.kp} \n\n Descriptors: \n{self.des}\n\n")

    def save_to_npz(self, filename: str) -> None:
        ''' Save model attributes to a .npz file '''
        keypoints = [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle, 'response': kp.response,
                      'octave': kp.octave, 'class_id': kp.class_id} for kp in self.kp]
        camera_params = {"mtx": self.camera_params["mtx"], "dist": self.camera_params["dist"],
                         "rvecs": self.camera_params["rvecs"], "tvecs": self.camera_params["tvecs"]}

        np.savez(filename, output_path=self.output_path, height=self.height, width=self.width,
                 kp=keypoints, des=self.des, vol=self.vol, camera_params=camera_params,
                 method=self.method)

    @classmethod
    def load(cls, filename: str) -> 'Model':
        ''' Load model attributes from a .npz file and create a Model instance '''
        data = np.load(filename, allow_pickle=True)

        if 'output_path' not in data:
            raise ValueError("The file does not contain the 'output_path' attribute.")

        keypoints = [cv.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'],
                                 kp['response'], kp['octave'], kp['class_id'])
                     for kp in data['kp']]

        camera_params = data['camera_params'].item()
        mtx = camera_params.get('mtx', None)
        dist = camera_params.get('dist', None)
        rvecs = camera_params.get('rvecs', None)
        tvecs = camera_params.get('tvecs', None)
        camera_params_dict = {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}

        new_object = cls(
            img=None,
            output_path=str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                            else data['output_path']),
            height=data['height'].item() if 'height' in data else None,
            width=data['width'].item() if 'width' in data else None,
            kp=keypoints,
            des=data['des'] if 'des' in data else None,
            vol=data['vol'] if 'vol' in data else None,
            camera_params=camera_params_dict,
            method=data['method'] if 'method' in data else None
        )
        output_path = str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                          else data['output_path'])
        new_object.upload_image(output_path, output_path)
        return new_object


def register(
    camera_params: str,
    input_image: str,
    vol: str,
    output_image: str,
    crop_method: str,
    points: list[str] = None,
    feature_method: str = "ORB",
    model_output: str = "model.npz"
):
    '''
    Register an image by loading camera parameters, cropping the image using the specified method,
    detecting features, and saving the model to a file.

    Parameters:
        camera_params (str): Path to camera parameters file.
        input_image (str): Path to the input image to be registered.
        vol (str): The volume of the object (needed for 3d rectangle frame).
        output_image (str): Path where the registered image will be saved.
        crop_method (str): Cropping method ("clicks", "points", or "none").
        points (list[str], optional): List of 8 coordinates (x1 y1 x2 y2 x3 y3 x4 y4) for 'points' crop method.
        feature_method (str, optional): Feature detection method ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT").
        model_output (str, optional): Path to save the model parameters.
    '''
    model = Model()
    model.load_camera_params(camera_params)
    model.upload_image(input_image, output_image, int(vol))

    if crop_method == "clicks":
        model.crop_image_by_clicks()
    elif crop_method == "points":
        if points:
            point_values = [int(coord) for coord in points]
            if len(point_values) != 8:
                raise ValueError(
                    "Exactly 4 points (8 values: x1 y1 x2 y2 x3 y3 x4 y4) are required for 'points' crop method.")
            points_array = np.array(point_values, dtype=np.int32).reshape(4, 2)
            model.crop_image_by_points(points_array)
        else:
            print("No points provided for 'points' crop method.")
    elif crop_method == "none":
        print("Skipping image cropping as per the selected method.")

    model.register(feature_method)
    model.save_to_npz(model_output)
    print(f"Model saved to {model_output}")
