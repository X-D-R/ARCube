import argparse
from src.detection.detection import Detector
from src.utils.draw_functions import draw_contours_of_rectangle
from src.tracking.frame import track_frame


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (detect).'''

    parser = argparse.ArgumentParser(description="Registration and Detection")

    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for detection
    detect_parser = subparsers.add_parser('detect', help="Detect features in an image or video")

    detect_parser.add_argument('--model_input', type=str, required=True, help="Path to the saved model file")
    detect_parser.add_argument('--camera_params', type=str, help="Path to camera parameters file ")
    detect_parser.add_argument('--input_image', type=str, help="Path to input image for detection")
    detect_parser.add_argument('--input_video', type=str, help="Path to input video for detection")
    detect_parser.add_argument('--video', action='store_true', help="if you want to detect video,"
                                                                    "don't use if you want to detect photo")
    detect_parser.add_argument('--output_image', type=str, help="Path to output image after detection")
    detect_parser.add_argument('--output_video', type=str, help="Path to output video after detection")

    args = parser.parse_args()

    if args.command == 'detect':
        detector = set_detector(args.model_input, args.camera_params)

        if args.video:
            track_frame(detector, args.input_video, args.output_video)
        else:
            img_points, src_pts, dst_pts = detector.detect_path(args.input_image)
            draw_contours_of_rectangle(args.input_image, args.output_image, img_points)
    else:
        print("Invalid command. Use 'detect'.")


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
    # example of the detection
    detector = set_detector("../ModelParams/model_test.npz", "../CameraParams/CameraParams.npz", True)

    img_points, src_pts, dst_pts = detector.detect_path("../examples/images/new_book_check.png")
    draw_contours_of_rectangle("../examples/images/new_book_check.png",
                               "../OutputFiles/OutputImages/contours_drawn.png", img_points)
    track_frame(detector, "../new_book_check/new_book_video_main.mp4",
                "../OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4", 60, 30, (0, 0, 255))

    # or
    # parse_args_and_execute()
    '''
    python detect.py detect --model_input "../ModelParams/model_test.npz" --camera_params "../CameraParams/CameraParams.npz" --input_video "../new_book_check/new_book_video_main.mp4" --video --output_video "../OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4"
    '''
