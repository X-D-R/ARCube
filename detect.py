import argparse
import os.path
from src.detection.detection import Detector
from src.tracking.frame import track_frame
from src.utils.draw_functions import draw_contours_of_rectangle

MAIN_DIR = os.path.dirname(os.path.abspath("detect.py"))


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (detect).'''

    parser = argparse.ArgumentParser(description="Detection")

    parser.add_argument('--model', type=str, help="Path to the saved model file")
    parser.add_argument('--demo', action='store_true', help="Use the default object model from the repository")
    parser.add_argument('--camera_params', type=str, required=True, help="Path to camera parameters file ")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or video for detection")
    parser.add_argument('--video', action='store_true', help="if you want to detect video,"
                                                                    "don't use if you want to detect photo")
    parser.add_argument('--output', type=str, required=True, help="Path to output image or video after detection")

    args = parser.parse_args()

    if args.demo:
        detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"),
                                args.camera_params)
    else:
        detector = set_detector(args.model, args.camera_params)

    if args.video:
        print('detecting object on video')
        track_frame(detector, args.input, args.output)
    else:
        print('detecting object on photo')
        img_points, src_pts, dst_pts = detector.detect_path(args.input)
        draw_contours_of_rectangle(args.input, args.output, img_points)


def set_detector(model_params_file: str, camera_params_file: str, use_flann: bool = True) -> Detector:
    '''
    This function set detector using model params
    :param model_params_file: str, path to where model should be saved
    :param camera_params_file: str, path to saved Camera parameters
    :param use_flann: bool, use FLANN-based matching or not
    :return: Detector, object of Detector class
    '''

    detector = Detector()
    # detector.set_detector_by_model("CameraParams/CameraParams.npz", model, True)
    detector.set_detector(camera_params_file, model_params_file, use_flann)

    return detector


def detect_photo(detector: Detector, input_file: str, output_file: str):
    '''
    This function detects object on the photo.
    :param detector: Detector, object of Detector class
    :param input_file: str, path to image where we use detector
    :param output_file: str, path to store the image with detected photo
    :return: 
    '''
    img_points, src_pts, dst_pts = detector.detect_path(input_file)
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
    
    python detect.py --demo --camera_params "ExampleFiles/CameraParams/CameraParams.npz" --input "ExampleFiles/new_book_check/new_book_video_main.mp4" --video --output "ExampleFiles/OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4"
    '''