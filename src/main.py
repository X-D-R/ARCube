import numpy as np
import os.path
from detection.detection import Detector
from registration.rectangle_model import RectangleModel, register
#from tracking.frame import track_frame

MAIN_DIR = os.path.split(os.path.split(os.path.abspath("main.py"))[0])[0]


#def register_to_model(object_corners_3d: np.ndarray, input_image: str, output_image: str, register_output: str,
                      #crop_method: str = 'corner', feature_method: str = 'SIFT') -> None:
def register_to_model(object_corners_3d: np.ndarray, input_image: str, output_image: str, register_output: str,
                    feature_method: str = 'SIFT') -> None:
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
        #crop_method=crop_method,
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

    ], dtype="float32")  # example of object_corners_3d
    register_to_model(object_corners_3d, os.path.join(MAIN_DIR, "ExampleFiles\\new_book_check\\book_3.jpg"),
                      os.path.join(MAIN_DIR, "ExampleFiles\\OutputFiles\\OutputImages\\output_script_test.jpg"),
                      os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_script_test.npz"), "SIFT")
    set_model(os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_script_test.npz"),
              os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_test.npz"))

    #detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_test.npz"),
    #                        os.path.join(MAIN_DIR, "ExampleFiles\\CameraParams\\CameraParams.npz"), True)

    # img_points, src_pts, dst_pts = detector.detect_path("../examples/images/new_book_check.png")
    # draw_contours_of_rectangle("../examples/images/new_book_check.png", "../OutputFiles/OutputImages/contours_drawn.png", img_points)
    #track_frame(detector, os.path.join(MAIN_DIR, "ExampleFiles\\new_book_check\\new_book_video_main.mp4"),
                #os.path.join(MAIN_DIR,
                #             "ExampleFiles\\OutputFiles/OutputVideos\\new_book_video_main_result_new_color.mp4"),
                #60, 30, (0, 0, 255))

    # # or
    # parse_args_and_execute()
    '''
    python main.py register --input_image "../old_files/andrew photo video/reference messy.jpg" --output_image "../OutputFiles/OutputImages/output_script_test.jpg" --crop_method "corner" --points 0 0 0 13 0 0 13 20.5 0 0 20.5 0 --feature_method "ORB" --model_output "../ModelParams/model_script_test.npz"
    '''
