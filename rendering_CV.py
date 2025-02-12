import cv2
import numpy as np
import math
import os.path
from src.detection.detection import detect_pose, Detector
from detect import set_detector
MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))

class render_CV:
    def __init__(self, model=None):
        self.DEFAULT_COLOR = (0, 0, 0)

    def projection_matrix(self, camera_matrix, homography):
        """
        From the camera calibration matrix and the estimated homography
        compute the 3D projection matrix
        """
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(camera_matrix), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(camera_matrix, projection)

    '''def hex_to_rgb(self, hex_color):
        """
        Helper function to convert hex strings to RGB
        """
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))'''

    def render(self, img, obj, projection, h, w, color=False):
        """
        Render a loaded obj model into the current video frame
        """
        DEFAULT_COLOR = (0, 0, 0)
        vertices = obj.vertices
        scale_matrix = np.eye(3) * 3
        #h, w = model.shape

        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            #points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                img = cv2.fillConvexPoly(img, imgpts, self.DEFAULT_COLOR)
            '''else:
                color = render_CV.hex_to_rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(img, imgpts, color)'''


        return img

def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    # matrix of camera parameters (made up but works quite well for me)
    detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"),
                            "ExampleFiles/CameraParams/CameraParams.npz")
    rend = render_CV()
    camera_matrix = detector.camera_params['mtx']
    cap = cv2.VideoCapture("ExampleFiles/new_book_check/new_book_video_main.mp4")
    i = detector.registration_params['img']
    h,w, c = i.shape
    Images = []

    obj = OBJ(os.path.join(MAIN_DIR, 'box.obj'), swapyz=True)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)

        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        #dst = cv2.perspectiveTransform(pts, homography)

        frame = cv2.polylines(frame, [np.int32(img_points)], True, 255, 3, cv2.LINE_AA)
        #if homography is not None:
         #   try:
                # obtain 3D projection matrix from homography matrix and camera parameters
        projection = rend.projection_matrix(camera_matrix, homography)
                # project cube or model
        frame = rend.render(frame, obj, projection, h, w, False)
                # frame = render(frame, model, projection)
            #except:
             #   pass

        cv2.imshow('frame', frame)
        height, width, channels = frame.shape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        Images.append(frame)

    #height, width, channels = Images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('result.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()



main()