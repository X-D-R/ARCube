from registration import Model
from detection import Detector
import argparse
import numpy as np


def register(args):
    '''
     Register an image by loading camera parameters, cropping the image using the specified method,
     detecting features, and saving the model to a file.
     params:
         args : The arguments parsed from the command line. Expected arguments include:
               - camera_params: Path to camera parameters file.
               - input_image: Path to the input image to be registered.
               - output_image: Path where the registered image will be saved.
               - crop_method: Cropping method (clicks, points, or none).
               - points: List of 8 coordinates (x1 y1 x2 y2 x3 y3 x4 y4) if crop_method is 'points'.
               - feature_method: Method for feature detection (ORB, KAZE, AKAZE, BRISK, SIFT).
               - model_output: Path to save the model parameters.
    '''
    model = Model()
    model.load_camera_params(args.camera_params)
    model.upload_image(args.input_image, args.output_image)

    if args.crop_method == "clicks":
        model.crop_image_by_clicks()
    elif args.crop_method == "points":
        if args.points:
            point_values = [int(coord) for coord in args.points]
            if len(point_values) != 8:
                raise ValueError(
                    "Exactly 4 points (8 values: x1 y1 x2 y2 x3 y3 x4 y4) are required for 'points' crop method.")
            points = np.array(point_values, dtype=np.int32).reshape(4, 2)
            model.crop_image_by_points(points)
        else:
            print("No points provided for 'points' crop method.")
    elif args.crop_method == "none":
        print("Skipping image cropping as per the selected method.")

    model.register(args.feature_method)

    model.save_to_npz(args.model_output)
    print(f"Model saved to {args.model_output}")


def detect(args):
    '''
    Detect features in an image or video.
    params:
        args : The arguments parsed from the command line. Expected arguments include:
              - model_input: Path to the saved model file.
              - camera_params: Path to the camera parameters file (optional for detection).
              - input_image: Path to the input image for detection (optional).
              - input_video: Path to the input video for detection (optional).
              - use_flann: Flag to use FLANN-based matching (optional).
              - draw_match: Flag to draw matches on the detected image (optional).
    '''
    model = Model.load(args.model_input)
    detector = Detector()
    detector.get_model_params(model)

    if args.camera_params:
        detector.load_camera_params(args.camera_params)

    detector.instance_method(args.use_flann)

    if args.input_image:
        detector.detect_image(args.input_image, useFlann=args.use_flann, drawMatch=args.draw_match)
    elif args.input_video:
        detector.detect_video(args.input_video)
    else:
        print("No input image or video provided for detection.")


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register or detect)'''

    parser = argparse.ArgumentParser(description="Registration and Detection")

    subparsers = parser.add_subparsers(dest="command")

    # Подкоманда для регистрации
    register_parser = subparsers.add_parser('register', help="Register an image")
    register_parser.add_argument('--camera_params', type=str, required=True, help="Path to camera parameters file")
    register_parser.add_argument('--input_image', type=str, required=True, help="Path to input image for registration")
    register_parser.add_argument('--output_image', type=str, required=True, help="Path to save the registered image")
    register_parser.add_argument('--crop_method', type=str, choices=['clicks', 'points', 'none'], required=True,
                                 help="Method for cropping the image")
    register_parser.add_argument('--points', type=str, nargs='*',
                                 help="Points for cropping (format: x1 y1 x2 y2 x3 y3 x4 y4),"
                                      " required if crop_method is 'points'")
    register_parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                                 required=True, help="Feature detection method")
    register_parser.add_argument('--model_output', type=str, required=True, help="Path to save the model parameters")

    # Подкоманда для детектирования
    detect_parser = subparsers.add_parser('detect', help="Detect features in an image or video")
    detect_parser.add_argument('--model_input', type=str, required=True, help="Path to the saved model file")
    detect_parser.add_argument('--camera_params', type=str,
                               help="Path to camera parameters file (optional, used for detection)")
    detect_parser.add_argument('--input_image', type=str, help="Path to input image for detection")
    detect_parser.add_argument('--input_video', type=str, help="Path to input video for detection")
    detect_parser.add_argument('--use_flann', action='store_true', help="Use FLANN-based matching")
    detect_parser.add_argument('--draw_match', action='store_true', help="Draw matches on the detected image/video")


    args = parser.parse_args()
    if args.command == 'register':
        register(args)
    elif args.command == 'detect':
        detect(args)
    else:
        print("Invalid command. Use 'register' or 'detect'.")
