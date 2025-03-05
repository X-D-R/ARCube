import argparse
import numpy as np
import os.path
from src.registration.rectangle_model import RectangleModel, register

MAIN_DIR = os.path.dirname(os.path.abspath("register.py"))


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register).'''

    parser = argparse.ArgumentParser(description="Registration")

    parser.add_argument('--input_image', type=str, required=False,
                                 help="Path to input image for registration")
    parser.add_argument('--output_image', type=str, required=True,
                                 help="Path to save the registered image")
    #parser.add_argument('--crop_method', type=str, choices=['manual', 'corner'], required=True,
                                 #help="Method for cropping the image ('manual' or 'corner')")
    parser.add_argument('--w', type=float, required=True,
                        help="Width of object ")
    parser.add_argument('--h', type=float, required=True,
                        help="Height of object ")
    parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                                 default="ORB", help="Feature detection method (default='ORB')")
    parser.add_argument('--model_output', type=str, required=True,
                                 help="Path to save the model parameters")
    parser.add_argument('--webcam', action='store_true', help="if you want to register an object using your webcam")

    args = parser.parse_args()

    w, h = args.w, args.h
    points = [0, 0, 0, w, 0, 0, w, h, 0, 0, h, 0]

    if args.input_image is None and args.webcam is None:
        raise ValueError("Provide input image or use webcam")

    if w is None or h is None:
        raise ValueError("Provide w and h")

    object_corners_3d = np.array(points, dtype="float32").reshape(-1, 3)
    #register_to_model(object_corners_3d, args.input_image, args.output_image, args.model_output, args.crop_method,
    #                  args.feature_method)
    if args.webcam:
        input_image = ''
        register_to_model(object_corners_3d, input_image, args.output_image, args.model_output,
                          args.feature_method, args.webcam)
    else:
        register_to_model(object_corners_3d, args.input_image, args.output_image, args.model_output,
                          args.feature_method)

    model = RectangleModel.load(args.model_output)
    print(model)



def register_to_model(object_corners_3d: np.ndarray, input_image: str, output_image: str, register_output: str,
                      feature_method: str = 'SIFT', webcam: bool = False) -> None:
    '''
    This function register object on given image
    :param object_corners_3d: np.ndarray, 3d coordinate points of object, dtype == np.float32
    :param input_image: str, path to original image of object
    :param output_image: str, path to where debug image should be saved
    :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
    :param feature_method: str, the method of registration, base - SIFT
    :param register_output: str, the path to where registration parameters should be saved in .npz format
    :param webcam: bool = False, flag if user want to register object by webcam
    :return: None
    '''
    register(
        input_image=input_image,
        output_image=output_image,
        object_corners_3d=object_corners_3d,
        #crop_method=crop_method,
        feature_method=feature_method,
        model_output=register_output,
        webcam=webcam
    )


if __name__ == "__main__":
    parse_args_and_execute()

    '''
    python register.py --input_image "ExampleFiles/new_book_check/book_3.jpg" --output_image "ExampleFiles/OutputFiles/OutputImages/output_script_test.jpg" --w 0.14 --h 0.21 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_test.npz"
    python register.py --output_image "exampleFiles/OutputFiles/OutputImages/output_varior_book.png" --w 0.13 --h 0.205 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_varior_book.npz" --webcam
    python register.py --input_image "ExampleFiles/examples/images/varior_book_iphone.jpg" --output_image "ExampleFiles/OutputFiles/OutputImages/varior_book_iphone.jpg" --w 0.135 --h 0.205 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_varior_book_iphone.npz"
    '''

