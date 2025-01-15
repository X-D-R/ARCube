import argparse
import numpy as np
from src.registration.rectangle_model import RectangleModel, register
from src.detection.detection import Detector
from src.utils.draw_functions import draw_contours_of_rectangle
from src.tracking.frame import track_frame


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register or detect).'''

    parser = argparse.ArgumentParser(description="Registration and Detection")

    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for registration
    register_parser = subparsers.add_parser('register', help="Register an image")

    register_parser.add_argument('--input_image', type=str, required=True,
                                 help="Path to input image for registration")
    register_parser.add_argument('--output_image', type=str, required=True,
                                 help="Path to save the registered image")
    register_parser.add_argument('--crop_method', type=str, choices=['photo', 'corner'], required=True,
                                 help="Method for cropping the image ('photo' or 'corner')")
    register_parser.add_argument('--points', type=float, nargs='+', required=True,
                                 help="List of 3D points (x1 y1 z1 x2 y2 z2 ... xn yn zn)")
    register_parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                                 default="ORB", help="Feature detection method (default='ORB')")
    register_parser.add_argument('--model_output', type=str, required=True,
                                 help="Path to save the model parameters")

    # # Subcommand for detection
    # detect_parser = subparsers.add_parser('detect', help="Detect features in an image or video")
    #
    # detect_parser.add_argument('--model_input', type=str, required=True, help="Path to the saved model file")
    # detect_parser.add_argument('--camera_params', type=str, help="Path to camera parameters file (optional)")
    # detect_parser.add_argument('--input_image', type=str, help="Path to input image for detection")
    # detect_parser.add_argument('--input_video', type=str, help="Path to input video for detection")
    # detect_parser.add_argument('--use_flann', action='store_true', help="Use FLANN-based matching")
    # detect_parser.add_argument('--draw_match', action='store_true', help="Draw matches on the detected image/video")

    args = parser.parse_args()

    if args.command == 'register':
        if args.points is None or len(args.points) % 3 != 0 or len(args.points) < 12:
            raise ValueError("Provide 3D points as a flat list of coordinates (x1 y1 z1 ... xn yn zn), and ensure "
                             "the total count is a multiple of 3.")

        object_corners_3d = np.array(args.points, dtype="float32").reshape(-1, 3)
        register(
            input_image=args.input_image,
            output_image=args.output_image,
            object_corners_3d=object_corners_3d,
            crop_method=args.crop_method,
            feature_method=args.feature_method,
            model_output=args.model_output
        )
    # elif args.command == 'detect':
    #     detect(
    #         model_input=args.model_input,
    #         camera_params=args.camera_params,
    #         input_image=args.input_image,
    #         input_video=args.input_video,
    #         use_flann=args.use_flann if args.use_flann is not None else False,
    #         draw_match=args.draw_match if args.draw_match is not None else False
    #     )
    # else:
    #     print("Invalid command. Use 'register' or 'detect'.")


def register_to_model(object_corners_3d: np.ndarray, input_image: str, output_image: str, register_output: str, crop_method: str = 'corner', feature_method: str = 'SIFT') -> None:
    '''
    This function register object on given image
    :param object_corners_3d: np.ndarray, 3d coordinate points of object, dtype == np.float32
    :param input_image: str, path to original image of object
    :param output_image: str, path to where debug image should be saved
    :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'photo' (place points directly on image)
    :param feature_method: str, the method of registration, base - SIFT
    :param register_output: str, the path to where registration parameters should be saved in .npz format
    :return: None
    '''
    register(
        input_image=input_image,
        output_image=output_image,
        object_corners_3d=object_corners_3d,
        crop_method=crop_method,
        feature_method=feature_method,
        model_output=register_output
    )

def set_model(register_path: str, model_output: str) -> None:
    '''
    This function set model using registration params
    :param register_path: str, path to saved registration parameters
    :param model_output: str, path to where model should be saved
    :return: None
    '''
    model = RectangleModel.load(register_path)
    model.save_to_npz(model_output)
    print(model)

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


if __name__ == "__main__":
    # example
    object_corners_3d = np.array([
        [0, 0, 0],  # Top-left
        [0.13, 0, 0],  # Top-right
        [0.13, 0.205, 0],  # Bottom-right
        [0, 0.205, 0],  # Bottom-left

    ], dtype="float32")
    '''
    register(
        input_image="old_files/andrew photo video/reference messy.jpg",
        output_image="OutputFiles/OutputImages/output_script_test.jpg",
        object_corners_3d=object_corners_3d,
        crop_method='corner', # or use crop_method='photo',
        feature_method="ORB",
        model_output="ModelParams/model_script_test.npz"
    )
    model = RectangleModel.load("ModelParams/model_script_test.npz")
    print(model)
    detector = Detector()
    detector.set_model("CameraParams/CameraParams.npz")
    detector.detect("")
    '''

    object_corners_3d = np.array([
        [0, 0, 0],  # Top-left
        [0.14, 0, 0],  # Top-right
        [0.14, 0.21, 0],  # Bottom-right
        [0, 0.21, 0],  # Bottom-left

    ], dtype="float32") # example of object_corners_3d
    register_to_model(object_corners_3d, "../new_book_check/book_3.jpg", "../OutputFiles/OutputImages/output_script_test.jpg", "../ModelParams/model_script_test.npz", 'corner', "SIFT")
    set_model("../ModelParams/model_script_test.npz", "../ModelParams/model_test.npz")

    detector = set_detector("../ModelParams/model_test.npz", "../CameraParams/CameraParams.npz", True)

    # img_points, src_pts, dst_pts = detector.detect_path("../examples/images/new_book_check.png")
    # draw_contours_of_rectangle("../examples/images/new_book_check.png", "../OutputFiles/OutputImages/contours_drawn.png", img_points)
    track_frame(detector, "../new_book_check/new_book_video_main.mp4", "../OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4", 60, 30, (0, 0, 255))

    # # or
    # parse_args_and_execute()
    '''
    python main.py register --input_image "../old_files/andrew photo video/reference messy.jpg" --output_image "../OutputFiles/OutputImages/output_script_test.jpg" --crop_method "corner" --points 0 0 0 13 0 0 13 20.5 0 0 20.5 0 --feature_method "ORB" --model_output "../ModelParams/model_script_test.npz"
    '''
