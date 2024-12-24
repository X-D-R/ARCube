import cv2 as cv
import numpy as np


class Model():
    '''
    Model class to perform image registration.
    '''

    def __init__(self, img: np.ndarray = None, output_path: str = '', height: int = 0, width: int = 0, kp: list = None,
                 des: np.ndarray = np.empty((0, 0)), vol: int = 0, camera_params: dict = None, method: str = '',
                 points_2d_3d: list = None, object_corners_2d: list = None, object_corners_3d: list = None):
        '''
        Initializes the Model class with the provided parameters.

        :param img: np.ndarray, the image data
        :param output_path: str, path to save the processed image
        :param height: int, height of the image
        :param width: int, width of the image
        :param kp: list, keypoints detected in the image
        :param des: np.ndarray, descriptors for the keypoints
        :param vol: int, volume of the object (if applicable)
        :param camera_params: dict, camera intrinsic parameters
        :param method: str, feature detection method used
        :param points_2d_3d: list, list of 2D-3D key point pairs
        :param object_corners_2d: list, 2D object corners for registration
        :param object_corners_3d: list, 3D object corners for registration
        '''
        self.img = img
        self.output_path = output_path
        self.height = height
        self.width = width
        self.kp = kp
        self.des = des
        self.vol = vol
        self.camera_params = camera_params
        self.method = method
        self.points_2d_3d = points_2d_3d if points_2d_3d is not None else []  # Инициализируем как пустой список
        self.object_corners_2d = object_corners_2d  # Object corners in 2D
        self.object_corners_3d = object_corners_3d  # Object corners in 3D


    def load_camera_params(self, path: str) -> None:
        '''
        Loads the camera parameters from a .npz file.

        :param path: str, path to the .npz file containing camera parameters (mtx, dist)
        :return: None
        '''
        if path.endswith('.npz'):
            with np.load(path) as file:
                self.camera_params = {
                    "mtx": file['mtx'],
                    "dist": file['dist']
                }
        else:
            print('Error: it is not .npz file')

    def upload_image(self, input_path: str, output_path: str, vol: int = 0) -> None:
        '''
        Loads the image of the object.

        :param input_path: str, path to the image file
        :param output_path: str, path to save the processed image
        :param vol: int, the volume of the object
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
        Detects keypoints and computes descriptors using the specified feature detection method.

        :param feature: str, the feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
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

    def get_params(self) -> tuple[list, np.ndarray]:
        return self.kp, self.des

    def register_point(self, point_2d: list, point_3d: list) -> None:
        '''
        Registers a single 2D-3D point pair.

        :param point_2d: list, 2D coordinates of the point
        :param point_3d: list, 3D coordinates of the point
        :return: None
        '''
        self.points_2d_3d.append({'2d': point_2d, '3d': point_3d})
        print(f"Registered 2D point {point_2d} with 3D point {point_3d}")

    def select_object_corners(self, crop_method: str) -> None:
        '''
        Allows the user to select the object corners interactively based on the crop method.

        :param crop_method: str, the method for selecting corners ("photo" or "corner")
        :return: None
        '''
        points_2d = []
        h = self.height
        w = self.width
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


    def save_to_npz(self, filename: str) -> None:
        '''
        Saves the model's attributes to a .npz file.

        :param filename: str, path to save the .npz file
        :return: None
        '''
        keypoints = [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle, 'response': kp.response,
                      'octave': kp.octave, 'class_id': kp.class_id} for kp in self.kp]
        camera_params = {"mtx": self.camera_params["mtx"], "dist": self.camera_params["dist"]}

        np.savez(filename, output_path=self.output_path, height=self.height, width=self.width,
                 kp=keypoints, des=self.des, vol=self.vol, camera_params=camera_params,
                 method=self.method, points_2d_3d=self.points_2d_3d,
                 object_corners_2d=self.object_corners_2d, object_corners_3d=self.object_corners_3d)

    @classmethod
    def load(cls, filename: str) -> 'Model':
        '''
        Loads a model from a .npz file.

        :param filename: str, path to the .npz file
        :return: Model, an instance of the Model class
        '''
        data = np.load(filename, allow_pickle=True)

        if 'output_path' not in data:
            raise ValueError("The file does not contain the 'output_path' attribute.")

        keypoints = [cv.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'],
                                 kp['response'], kp['octave'], kp['class_id'])
                     for kp in data['kp']]

        camera_params = data['camera_params'].item()
        mtx = camera_params.get('mtx', None)
        dist = camera_params.get('dist', None)
        camera_params_dict = {"mtx": mtx, "dist": dist}

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
            method=data['method'] if 'method' in data else None,
            points_2d_3d=data['points_2d_3d'],
            object_corners_2d=data['object_corners_2d'],
            object_corners_3d=data['object_corners_3d']
        )
        output_path = str(data['output_path'].item() if isinstance(data['output_path'], np.ndarray)
                          else data['output_path'])
        new_object.upload_image(output_path, output_path)
        return new_object

    def register_with_object_corners(self, feature_method: str, object_corners_3d: np.ndarray,
                                     crop_method: str = 'photo') -> None:
        '''
        Performs registration, including interactive corner selection.

        :param feature_method: str, the feature detection method to use (e.g., "ORB", "SIFT")
        :param object_corners_3d: np.ndarray, the 3D coordinates of the object corners
        :param crop_method: str, the method used for cropping ("photo" or "corner")
        :return: None
        '''
        self.select_object_corners(crop_method)
        self.object_corners_3d = object_corners_3d
        self.register(feature_method)
        self.map_keypoints_to_3d_plane()
        print("Registration completed with object corners.")


def register(camera_params: str, input_image: str, output_image: str, vol: int,
             object_corners_3d: np.ndarray, crop_method: str, feature_method: str, model_output: str) -> None:

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
    model = Model()
    model.load_camera_params(camera_params)
    model.upload_image(input_image, output_image, vol)
    model.register_with_object_corners(feature_method, object_corners_3d, crop_method)
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

