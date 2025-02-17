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


    '''def hex_to_rgb(self, hex_color):
        """
        Helper function to convert hex strings to RGB
        """
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))'''

    def render(self, img, obj, projection, rvecs, tvec, mtx, dst, color=False):
        """
        Render a loaded obj model into the current video frame
        """
        DEFAULT_COLOR = (0, 0, 0)
        vertices = obj.vertices


        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            #dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            dst, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvecs, tvec, mtx, dst)
            print(dst)
            imgpts = np.int32(dst)

            if color is False:
                img = cv2.fillConvexPoly(img, imgpts, self.DEFAULT_COLOR)


        return img

def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"),
                            "ExampleFiles/CameraParams/CameraParams.npz")
    rend = render_CV()
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']
    cap = cv2.VideoCapture("ExampleFiles/new_book_check/new_book_video_main.mp4")
    i = detector.registration_params['img']
    h,w, c = i.shape
    Images = []

    obj = OBJ(os.path.join(MAIN_DIR, 'box_2.obj'), swapyz=True)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)


        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        #dst = cv2.perspectiveTransform(pts, homography)
        if img_points is not None:

            frame = cv2.polylines(frame, [np.int32(img_points)], True, 255, 3, cv2.LINE_AA)
            valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
            if valid:

                projection = camera_matrix @ np.hstack((rvecs, tvec))
                frame = rend.render(frame, obj, projection, rvecs, tvec, camera_matrix, dist_coeffs, False)

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