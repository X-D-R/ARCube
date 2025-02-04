import argparse
import numpy as np
import os.path
from src.registration.rectangle_model import RectangleModel, register

MAIN_DIR = os.path.dirname(os.path.abspath("register.py"))


def parse_args_and_execute():
    '''Parse command-line arguments and execute the appropriate function (register).'''

    parser = argparse.ArgumentParser(description="Registration")

    parser.add_argument('--input_image', type=str, required=True,
                                 help="Path to input image for registration")
    parser.add_argument('--output_image', type=str, required=True,
                                 help="Path to save the registered image")
    #parser.add_argument('--crop_method', type=str, choices=['manual', 'corner'], required=True,
                                 #help="Method for cropping the image ('manual' or 'corner')")
    parser.add_argument('--points', type=float, nargs='+', required=True,
                                 help="List of 3D points (x1 y1 z1 x2 y2 z2 ... xn yn zn). "
                                      "Default is 4 points or 12 coordinates")
    parser.add_argument('--feature_method', type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"],
                                 default="ORB", help="Feature detection method (default='ORB')")
    parser.add_argument('--model_output', type=str, required=True,
                                 help="Path to save the model parameters")

    args = parser.parse_args()

    if args.points is None or len(args.points) % 3 != 0 or len(args.points) < 12:
        raise ValueError("Provide 3D points as a flat list of coordinates (x1 y1 z1 ... xn yn zn), and ensure "
                         "the total count is a multiple of 3.")

    object_corners_3d = np.array(args.points, dtype="float32").reshape(-1, 3)
    #register_to_model(object_corners_3d, args.input_image, args.output_image, args.model_output, args.crop_method,
    #                  args.feature_method)
    register_to_model(object_corners_3d, args.input_image, args.output_image, args.model_output,
                      args.feature_method)
    model = RectangleModel.load(args.model_output)
    print(model)



def register_to_model(object_corners_3d: np.ndarray, input_image: str, output_image: str, register_output: str,
                      feature_method: str = 'SIFT') -> None:
    '''
    This function register object on given image
    :param object_corners_3d: np.ndarray, 3d coordinate points of object, dtype == np.float32
    :param input_image: str, path to original image of object
    :param output_image: str, path to where debug image should be saved
    :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
    :param feature_method: str, the method of registration, base - SIFT
    :param register_output: str, the path to where registration parameters should be saved in .npz format
    :return: None
    '''
    register(
        input_image=input_image,
        output_image=output_image,
        object_corners_3d=object_corners_3d,
        #crop_method=crop_method,
        feature_method=feature_method,
        model_output=register_output
    )


if __name__ == "__main__":
    # example of the object registration
    # object_corners_3d = np.array([
    #     [0, 0, 0],  # Top-left
    #     [0.14, 0, 0],  # Top-right
    #     [0.14, 0.21, 0],  # Bottom-right
    #     [0, 0.21, 0],  # Bottom-left
    #
    # ], dtype="float32")  # example of object_corners_3d
    # print(os.path.join(MAIN_DIR, "ExampleFiles", "new_book_check", "book_3.jpg"))
    # register_to_model(object_corners_3d, os.path.join(MAIN_DIR, "ExampleFiles", "new_book_check", "book_3.jpg"),
    #                   os.path.join(MAIN_DIR, "ExampleFiles", "OutputFiles", "OutputImages", "output_script_test.jpg"),
    #                   os.path.join(MAIN_DIR, "ExampleFiles", "ModelParams", "model_test.npz"), 'corner', "SIFT")

    # or
    parse_args_and_execute()
    '''
    python register.py --input_image "ExampleFiles/new_book_check/book_3.jpg" --output_image "ExampleFiles/OutputFiles/OutputImages/output_script_test.jpg" --points 0 0 0 0.14 0 0 0.14 0.21 0 0 0.21 0 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_test.npz"

    '''
