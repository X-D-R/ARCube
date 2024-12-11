import argparse
import numpy as np
from detection import Detector, detect
from registration import Model, register



def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register or detect).'''

    parser = argparse.ArgumentParser(description="Registration and Detection")

    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for registration
    register_parser = subparsers.add_parser('register', help="Register an image")

    # Subcommand for registration
    register_parser.add_argument('--camera_params', type=str, required=True,
                        help="Path to camera parameters file (.npz)")
    register_parser.add_argument('--input_image', type=str, required=True,
                        help="Path to input image for registration")
    register_parser.add_argument('--output_image', type=str, required=True,
                        help="Path to save the registered image")
    register_parser.add_argument('--vol', type=int, required=False, default=0,
                        help="Volume of the object (optional, default=0)")
    register_parser.add_argument('--crop_method', type=str, choices=['photo', 'corner'], required=True,
                        help="Method for cropping the image ('photo' or 'corner')")
    register_parser.add_argument('--points', type=float, nargs='+', required=True,
                        help="List of 3D points (x1 y1 z1 x2 y2 z2 ... xn yn zn)")
    register_parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                        default="ORB", help="Feature detection method (default='ORB')")
    register_parser.add_argument('--model_output', type=str, required=True,
                        help="Path to save the model parameters")

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
        if args.points is None or len(args.points) % 3 != 0 or len(args.points) < 12:
            raise ValueError("Provide 3D points as a flat list of coordinates (x1 y1 z1 ... xn yn zn), and ensure "
                             "the total count is a multiple of 3.")

        object_corners_3d = np.array(args.points, dtype="float32").reshape(-1, 3)
        register(
            camera_params=args.camera_params,
            input_image=args.input_image,
            output_image=args.output_image,
            vol=args.vol,
            object_corners_3d=object_corners_3d,
            crop_method=args.crop_method,
            feature_method=args.feature_method,
            model_output=args.model_output
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
    ''' An example of how to register and detect object '''

    # Example of 3D object coordinates
    object_corners_3d = np.array([
        [0, 0, 0],  # Top-left corner
        [13, 0, 0],  # Top-right corner
        [13, 20.5, 0],  # Bottom-right corner
        [0, 20.5, 0],  # Bottom-left corner
    ], dtype="float32")

    # Parameters for registration
    camera_params_path = "CameraParams/cam_params_andrew.npz"
    input_image_path = "old_files/andrew photo video/reference messy.jpg"
    output_image_path = "output_script_test.jpg"
    model_output_path = "model_script_test.npz"

    # Calling the registration function
    register(
        camera_params=camera_params_path,
        input_image=input_image_path,
        output_image=output_image_path,
        vol=0,  # Object volume (if required)
        object_corners_3d=object_corners_3d,
        crop_method='corner',  # Using corner method
        feature_method="ORB",  # Detection method
        model_output=model_output_path  # Saving model parameters
    )
    print("Registration completed successfully.")


    model2=Model.load(output_image_path) # Loading model parameters from a file

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
    '''
    # new
    python main.py register --camera_params "CameraParams/cam_params_andrew.npz" --input_image "old_files/andrew photo video/reference messy.jpg" --output_image "output_script_test.jpg" --crop_method "corner" --points 0 0 0 13 0 0 13 20.5 0 0 20.5 0 --feature_method "ORB" --model_output "model_script_test.npz"
    '''
    # old
    # python main.py register --camera_params "CameraParams/CameraParams.npz" --input_image "old_files/andrew photo video/reference messy.jpg" --output_image "output_script_test.jpg" --crop_method "none" --feature_method "SIFT" --model_output "model_script_test.npz"
    # python main.py detect --model_input "model_script_test.npz" --input_image "old_files/andrew photo video/second pic messy.jpg" --use_flann --draw_match
    # or
    # test_handy_register_detect()


