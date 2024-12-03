import argparse
from registration import Model, register
from detection import Detector, detect


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register or detect).'''

    parser = argparse.ArgumentParser(description="Registration and Detection")

    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for registration
    register_parser = subparsers.add_parser('register', help="Register an image")
    register_parser.add_argument('--camera_params', type=str, required=True, help="Path to camera parameters file")
    register_parser.add_argument('--input_image', type=str, required=True, help="Path to input image for registration")
    register_parser.add_argument('--vol', type=str, help="Volume of the object")
    register_parser.add_argument('--output_image', type=str, required=True, help="Path to save the registered image")
    register_parser.add_argument('--crop_method', type=str, choices=['clicks', 'points', 'none'], required=True,
                                 help="Method for cropping the image")
    register_parser.add_argument('--points', type=str, nargs='*',
                                 help="Points for cropping (format: x1 y1 x2 y2 x3 y3 x4 y4),"
                                      " required if crop_method is 'points'")
    register_parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                                 help="Feature detection method", default="ORB")
    register_parser.add_argument('--model_output', type=str, help="Path to save the model parameters",
                                 default="model.npz")

    # Subcommand for detection
    detect_parser = subparsers.add_parser('detect', help="Detect features in an image or video")
    detect_parser.add_argument('--model_input', type=str, required=True, help="Path to the saved model file")
    detect_parser.add_argument('--camera_params', type=str, help="Path to camera parameters file (optional)")
    detect_parser.add_argument('--input_image', type=str, help="Path to input image for detection")
    detect_parser.add_argument('--input_video', type=str, help="Path to input video for detection")
    detect_parser.add_argument('--use_flann', action='store_true', help="Use FLANN-based matching")
    detect_parser.add_argument('--draw_match', action='store_true', help="Draw matches on the detected image/video")
    args = parser.parse_args()
    if args.command == 'register':
        register(
            camera_params=args.camera_params,
            input_image=args.input_image,
            output_image=args.output_image,
            crop_method=args.crop_method,
            vol=args.vol if args.vol is not None else '0',
            points=args.points,
            feature_method=args.feature_method if args.feature_method is not None else "ORB",
            model_output=args.model_output if args.model_output is not None else "model.npz"
        )
    elif args.command == 'detect':
        detect(
            model_input=args.model_input,
            camera_params=args.camera_params,
            input_image=args.input_image,
            input_video=args.input_video,
            use_flann=args.use_flann if args.use_flann is not None else False,
            draw_match=args.draw_match if args.draw_match is not None else False
        )
    else:
        print("Invalid command. Use 'register' or 'detect'.")


def test_handy_register_detect():
    # Test photo by different detection methods
    # model_check = Model()
    # cam_params_path = "./CameraParams/CameraParams.npz"
    # input_img_path = "./old_files/DanielFiles/book.jpg"
    # output_img_path = "check_model.jpg"
    # model_check._check(cam_params_path, input_img_path, output_img_path)

    # Good example of how to register an object (photo)
    model = Model()
    model.load_camera_params("./CameraParams/CameraParams.npz")
    model.upload_image('old_files/andrew photo video/messy krivoy.jpg', 'test_model.jpg')
    model.crop_image_by_clicks()
    model.register('SIFT')

    # Saving and loading model params
    path = 'test_model.npz'
    model.save_to_npz(path)
    model2 = Model.load(path)

    # Good example of how to create a cls Detector object
    detector = Detector()
    detector.get_model_params(model2)
    detector.instance_method(True)
    detector.load_camera_params("./CameraParams/CameraParams.npz")

    # Detecting image on image/video
    detector.detect_image("old_files/andrew photo video/second pic messy.jpg", useFlann=True, drawMatch=True)
    # or
    # detector.detect_video("./examples/videos/book_video.mp4")


if __name__ == "__main__":
    parse_args_and_execute()
    # python main.py register --camera_params "CameraParams/CameraParams.npz" --input_image "old_files/andrew photo video/reference messy.jpg" --output_image "output_script_test.jpg" --crop_method "none" --feature_method "SIFT" --model_output "model_script_test.npz"
    # python main.py detect --model_input "model_script_test.npz" --input_image "old_files/andrew photo video/second pic messy.jpg" --use_flann --draw_match
    # or
    # test_handy_register_detect()

