import numpy as np
import cv2 as cv
import pickle


class Model():

    def __init__(self, img: np.ndarray = None, height: int = 0, width: int = 0, kp: list = None,
                 des: np.ndarray = np.empty((0, 0)), vol: int = 0, camera_params: dict = None, method: str = ''):
        '''
        Attributes:
        - img (np.ndarray): The image data
        - height (int): The height of the picture
        - width (int): The width of the picture
        - kp (list): Key points detected in the image
        - des (np.ndarray): Descriptors for the key points
        - vol (int): The volume of the object (needed for 3d rectangle frame)
        - camera_params (dict): Dictionary containing camera parameters
        - method (str): feature detection method to use ("ORB", "KAZE", "AKAZE", "BRISK", "SIFT")
        '''
        self.img: img
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

    def upload_image(self, path: str) -> None:
        '''
        This func should upload image using cv
        from given path and save to self.img
        :param path: str
        :return: None
        '''
        try:
            self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            self.height, self.width = self.img.shape
            self.vol = int(input("Input volume:"))
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


    def order_points(self,pts):
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
        image=self.img.copy()
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


    def _check(self, path_params: str, path_img: str) -> None:
        self.load_camera_params(path_params)
        self.upload_image(path_img)
        for feature in ["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"]:
            self.register(feature)
            print(f"Feature: {feature}\n\n")
            print(f" KeyPoints: \n {self.kp} \n\n Descriptors: \n{self.des}\n\n")

    def save_to_npz(self, filename: str) -> None:
        ''' Save model attributes to a .npz file '''
        keypoints = [{'pt': kp.pt, 'size': kp.size, 'angle': kp.angle, 'response': kp.response,
                      'octave': kp.octave, 'class_id': kp.class_id} for kp in self.kp]
        np.savez(filename, img=self.img, height=self.height, width=self.width,
                 kp=keypoints, des=self.des, vol=self.vol, camera_params=self.camera_params,
                 method=self.method)

        
    @classmethod
    def load(cls, filename: str) -> 'Model':
        ''' Load model attributes from a .npz file and create a Model instance '''
        data = np.load(filename, allow_pickle=True)

        if 'img' not in data:
            raise ValueError("The file does not contain the 'img' attribute.")

        keypoints = [cv.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'],
                                 kp['response'], kp['octave'], kp['class_id'])
                     for kp in data['kp']]

        return cls(
            img=data['img'],  # Ensuring 'img' is loaded correctly
            height=data['height'].item() if 'height' in data else None,
            width=data['width'].item() if 'width' in data else None,
            kp=keypoints,
            des=data['des'] if 'des' in data else None,
            vol=data['vol'] if 'vol' in data else None,
            camera_params=data['camera_params'].item() if 'camera_params' in data else None,
            method=data['method'] if 'method' in data else None
        )



#
#model = Model()
#model.load_camera_params("./CameraParams/CameraParams.npz")
#model.upload_image("./old_files/DanielFiles/book.jpg")
#model.register("ORB")
#model.save_to_npz("book_reg")
#model._check("./CameraParams/CameraParams.npz", "./old_files/DanielFiles/book.jpg")
'''
#model._check("./CameraParams/CameraParams.npz", "./old_files/andrew photo video/reference_messy_1.jpg")

model = Model()
model.upload_image('old_files/andrew photo video/messy krivoy.jpg')
model.load_camera_params('CameraParams/CameraParams.npz')
'''
points = np.array(((340, 230), (808, 368), (570, 1140), (92, 969)))
model.crop_image_by_points(points)
cv.waitKey(0)
cv.imshow('Cropped Image', model.img)
cv.waitKey(0)

model.crop_image_by_clicks()
cv.waitKey(0)
cv.imshow('Cropped Image2', model.img)
cv.waitKey(0)
'''

model.register('SIFT')

path='testmodel.npz'
model.save_to_npz(path)

model2=Model.load(path)
print(model2.kp,model2.des,model2.vol,model2.camera_params,model2.method,model2.height,model2.width)
cv.waitKey(0)
cv.imshow(' Image m2', model2.img)
cv.waitKey(0)
'''
#model._check("./CameraParams/CameraParams.npz", "./old_files/andrew photo video/reference_messy_1.jpg")
