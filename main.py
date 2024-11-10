from registration import Model
from detection import Detector
from parse_args import parse_args_and_execute

if __name__ == "__main__":
    parse_args_and_execute()
    # python main.py register --camera_params "CameraParams/CameraParams.npz" --input_image "old_files/andrew photo video/reference messy.jpg" --output_image "output_script_test.jpg" --crop_method "none" --feature_method "SIFT" --model_output "model_script_test.npz"
    # python main.py detect --model_input "model_script_test.npz" --input_image "old_files/andrew photo video/second pic messy.jpg" --use_flann --draw_match

    # or

    # # Test photo by different detection methods
    # # model_check = Model()
    # # cam_params_path = "./CameraParams/CameraParams.npz"
    # # input_img_path = "./old_files/DanielFiles/book.jpg"
    # # output_img_path = "check_model.jpg"
    # # model_check._check(cam_params_path, input_img_path, output_img_path)
    #
    # # Good example of how to register an object (photo)
    # model = Model()
    # model.load_camera_params("./CameraParams/CameraParams.npz")
    # model.upload_image('old_files/andrew photo video/messy krivoy.jpg', 'test_model.jpg')
    # model.crop_image_by_clicks()
    # model.register('SIFT')
    #
    # # Saving and loading model params
    # path = 'test_model.npz'
    # model.save_to_npz(path)
    # model2 = Model.load(path)
    #
    # # Good example of how to create a cls Detector object
    # detector = Detector()
    # detector.get_model_params(model2)
    # detector.instance_method(True)
    # detector.load_camera_params("./CameraParams/CameraParams.npz") #IDK is it needed if the model has params
    #
    # # Detecting image on image/video
    # detector.detect_image("old_files/andrew photo video/second pic messy.jpg", useFlann=True, drawMatch=True)
    # #or
    # #detector.detect_video("./examples/videos/book_video.mp4")
