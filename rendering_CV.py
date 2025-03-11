import cv2
import numpy as np
import os.path
from src.detection.detection import detect_pose, Detector
MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


def generate_obj_file(filename, width, height):
    with open(filename, 'w') as file:
        file.write("v 0 0 0\n")
        file.write(f"v {width} 0 0\n")
        file.write(f"v {width} 0 {height}\n")
        file.write(f"v 0  0 {height}\n")
        file.write("f 1 2 3 4\n")



class OBJ:
    def __init__(self, filename: str, swapyz=False):
        '''
        Load obj model
        :param filename: path to obj file
        :param swapyz:
        '''
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
                self.faces.append((face, norms, texcoords))

class render_CV:
    def __init__(self, model=None):
        self.DEFAULT_COLOR = (0, 0, 0)



    def render(self, img, obj, rvecs, tvec, mtx, dst, texture):
        '''
        Render a loaded obj model into the current video frame
        :param img: current frame
        :param obj: obj model of OBJ class
        :param rvecs: matrix of rotation
        :param tvec: vector of translation
        :param mtx: camera matrix
        :param dst: distortion coefficients of camera
        :param texture:
        return frame with superimposed texture
        '''
        vertices = obj.vertices
        h, w, _ = img.shape
        texture_h, texture_w, _ = texture.shape
        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            dst, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvecs, tvec, mtx, dst)
            dst[np.isnan(dst)] = 0
            imgpts = np.int32(dst)
            #img = cv2.polylines(img, [np.int32(imgpts)], True, 255, 3, cv2.LINE_AA)
            x_1 = imgpts[0][0][0]
            y_1 = imgpts[0][0][1]
            x_2 = imgpts[1][0][0]
            y_2 = imgpts[1][0][1]
            distant = np.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)
            distant = 0 if np.isnan(distant) else distant
            size_of_texture = int(distant // 2)
            if size_of_texture == 0:
                return None
            texture = cv2.resize(texture, (size_of_texture, size_of_texture))
            x_start = x_1 + abs(x_1 - imgpts[2][0][0]) // 2 - size_of_texture // 2
            y_start = y_1 + abs(y_1 - imgpts[2][0][1]) // 2 - size_of_texture // 2

            roi = img[y_start:y_start+size_of_texture, x_start:x_start+size_of_texture]

            texture2gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(texture2gray, 220, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            if mask.shape == roi.shape[:2] and mask.dtype == np.uint8:

                img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

                img2_fg = cv2.bitwise_and(texture, texture, mask=mask_inv)

                dst = cv2.add(img1_bg, img2_fg)

                if dst.shape == img[y_start:y_start + size_of_texture, x_start:x_start + size_of_texture].shape:
                    img[y_start:y_start + size_of_texture, x_start:x_start + size_of_texture] = dst


        return img

def rendering_video(detector, cap, obj_path):
    """
    This functions loads the target surface image,
    """


    rend = render_CV()
    camera_matrix = detector.camera_params['mtx']
    dist_coeffs = detector.camera_params['dist']
    #cap = cv2.VideoCapture("ExampleFiles/new_book_check/new_book_video_main.mp4")
    i = detector.registration_params['img']
    h,w, c = i.shape
    Images = []

    generate_obj_file("box.obj", width=0.14, height=0.21)

    obj = OBJ(os.path.join(os.path.join(MAIN_DIR, "ExampleFiles", "3d_models", "box_CV.npz")), swapyz=True)
    texture = cv2.imread('hse.jpg')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break
        img_points, inliers_original, inliers_frame, kp, good, homography, mask = detector.detect(frame)

        if img_points is not None:

            #frame = cv2.polylines(frame, [np.int32(img_points)], True, 255, 3, cv2.LINE_AA)
            valid, rvecs, tvec = detect_pose(inliers_frame, inliers_original, camera_matrix, dist_coeffs)
            if valid:

                frame = rend.render(frame, obj, rvecs, tvec, camera_matrix, dist_coeffs, texture)
        Images.append(frame)
        cv2.imshow('To stop running press "Escape"', frame)
        height, width, channels = frame.shape
        if cv2.waitKey(33) & 0xFF == 27:
            break

    #cv2.destroyAllWindows()
    # saving video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter('result_2.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    video.release()
    print("Saved video")
    cv2.destroyAllWindows()


    '''fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('result_2.mp4', fourcc, 30, (width, height))

    for i in range(len(Images)):
        video.write(Images[i])
        #if cv2.waitKey(33) & 0xFF == 27:
         #   break


    cap.release()

    cv2.destroyAllWindows()'''


