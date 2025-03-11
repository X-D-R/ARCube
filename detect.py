import argparse
import os.path
import numpy as np
import cv2 as cv
from src.detection.detection import Detector
from src.tracking.frame import track_frame, track_frame_cam
from src.utils.draw_functions import draw_contours_of_rectangle, visualize_matches_on_photo
from rendering_CV import rendering_video, generate_obj_file
MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (detect).'''

    parser = argparse.ArgumentParser(description="Detection")

    parser.add_argument('--model', type=str, help="Path to the saved model file")
    parser.add_argument('--demo', action='store_true', help="Use the default object model from the repository")
    parser.add_argument('--camera_params', type=str, required=False, help="Path to camera parameters file")
    parser.add_argument('--input', type=str, required=False, help="Path to input image or video for detection")
    parser.add_argument('--video', action='store_true', help="if you want to detect video,"
                                                                    "don't use if you want to detect photo")
    parser.add_argument('--output', type=str, required=True, help="Path to output image or video after detection")
    parser.add_argument('--use_tracker', action='store_true', help="Use if you want to use tracking")
    parser.add_argument('--web_camera', action='store_true', help="Use if you want to use your camera")
    parser.add_argument('--visualize_matches', action='store_true', help="Use if you want to visualize matches with the reference image")
    parser.add_argument('--render', action='store_true', help="Use if you want to use rendering")

    args = parser.parse_args()

    camera_params_approximate = {}
    if args.camera_params is None:
        imgSize = (0, 0)
        if args.video and not args.web_camera:
            cap = cv.VideoCapture(args.input)
            ret, frame = cap.read()
            imgSize = frame.shape
            cap.release()
        elif args.web_camera:
            cap = cv.VideoCapture(0)
            ret, frame = cap.read()
            imgSize = frame.shape
            cap.release()
        else:
            img = cv.imread(args.input)
            imgSize = img.shape
        w, h = imgSize[0], imgSize[1]
        f = 0.9*max(w, h)
        camera_params_approximate = {'mtx': np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], np.float32), 'dist': np.array([0, 0, 0, 0, 0], np.float32)}

    if args.demo:
        detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"),
                                args.camera_params, camera_params_approximate=camera_params_approximate)
    else:
        detector = set_detector(args.model, args.camera_params, camera_params_approximate=camera_params_approximate)

    if args.video:
        print('detecting object on video')
        track_length = 50 if args.use_tracker else 1
        if args.render:
            coord_3d = detector.registration_params['object_corners_3d']
            w = coord_3d[2][0]
            h = coord_3d[2][1]
            generate_obj_file(os.path.join(MAIN_DIR, "ExampleFiles", "3d_models", "box_CV.npz"), w, h)
            print("OBJ file is generated")

        if args.visualize_matches:
            print('visualizing matches with the reference image')
            if args.web_camera:
                track_frame_cam(detector, args.output, track_length=track_length, visualizing_matches=True)
            else:
                track_frame(detector, args.input, args.output, track_length=track_length, visualizing_matches=True)
        else:
            if args.web_camera:
                track_frame_cam(detector, args.output, track_length=track_length)
            else:
                track_frame(detector, args.input, args.output, track_length=track_length)

    else:
        print('detecting object on photo')
        img_points, src_pts, dst_pts, keypoints, matches = detector.detect_path(args.input)
        draw_contours_of_rectangle(args.input, args.output, img_points)

        if args.visualize_matches:
            print('visualizing matches with the reference image')
            visualize_matches_on_photo(detector.registration_params["img"], detector.registration_params["key_points_2d"],
                              args.input, keypoints, matches)


def set_detector(model_params_file: str, camera_params_file: str, use_flann: bool = True, camera_params_approximate: dict = {}) -> Detector:
    '''
    This function set detector using model params
    :param model_params_file: str, path to where model should be saved
    :param camera_params_file: str, path to saved Camera parameters
    :param use_flann: bool, use FLANN-based matching or not
    :return: Detector, object of Detector class
    '''

    detector = Detector()
    # detector.set_detector_by_model("CameraParams/CameraParams.npz", model, True)
    detector.set_detector(camera_params_file, model_params_file, use_flann, camera_params_approximate)

    return detector


def detect_photo(detector: Detector, input_file: str, output_file: str):
    '''
    This function detects object on the photo.
    :param detector: Detector, object of Detector class
    :param input_file: str, path to image where we use detector
    :param output_file: str, path to store the image with detected photo
    :return: 
    '''
    img_points, src_pts, dst_pts, keypoints, matches = detector.detect_path(input_file)
    draw_contours_of_rectangle(input_file, output_file, img_points)


if __name__ == "__main__":
    # example of the detection
    # detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"),
    #                         os.path.join(MAIN_DIR, "ExampleFiles", "CameraParams", "CameraParams.npz"), True)
    # # Photo detection
    # detect_photo(detector, os.path.join(MAIN_DIR, "ExampleFiles", "examples", "images", "new_book_check.png"),
    #              os.path.join(MAIN_DIR, "ExampleFiles", "OutputFiles", "OutputImages", "contours_drawn.png"))
    #
    # # Video detection
    # track_frame(detector, os.path.join(MAIN_DIR, "ExampleFiles", "new_book_check", "new_book_video_main.mp4"),
    #             os.path.join(MAIN_DIR,
    #                          "ExampleFiles", "OutputFiles", "OutputVideos", "new_book_video_main_result_new_color.mp4"), 60,
    #             30, (0, 0, 255))

    # or
    parse_args_and_execute()
    '''
    python detect.py --model "ExampleFiles/ModelParams/model_test.npz" --camera_params "ExampleFiles/CameraParams/CameraParams.npz" --input "ExampleFiles/new_book_check/new_book_video_main.mp4" --video --output "ExampleFiles/OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4"
    
    python detect.py --demo --camera_params "ExampleFiles/CameraParams/CameraParams.npz" --input "ExampleFiles/new_book_check/new_book_video_main.mp4" --video --output "ExampleFiles/OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4" --use_tracker
    
    python detect.py --model "ExampleFiles/ModelParams/model_varior_book_iphone.npz" --video --output "ExampleFiles/OutputFiles/OutputVideos/varior_book_result_iphone.mp4" --use_tracker --web_camera 
    '''