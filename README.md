# Object registration and detection using OpenCV

This project provides tools for registrating objects (images) and detecting in images and videos. 

## Table of Contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Scripts](#scripts)
6. [License](#license)

## Description

This project enables:
- Loading camera calibration parameters.
- Loading image, cropping it (through clicking on the object corners on the picture) and loading 3d parameters of the object .
- Registrating object from image for futher detection.
- Loading camera calibration parameters.
- Detecting an object in images and videos using algorithms: "ORB", "KAZE", "AKAZE", "BRISK", "SIFT".
- Saving detection results as image or video.
- Running registration and detection scripts

## Requirements

- contourpy==1.3.0
- cycler==0.12.1
- fonttools==4.54.1
- kiwisolver==1.4.7
- matplotlib==3.9.2
- numpy==2.1.3
- opencv-python==4.10.0.84
- packaging==24.1
- pillow==11.0.0
- pyparsing==3.2.0
- python-dateutil==2.9.0.post0
- six==1.16.0

You can install it using the `requirements.txt` file.

## Installation

1. Clone the repository:

   ```bash
   git clone <https://github.com/X-D-R/ARCube.git>
   cd <repository_folder>
   ```

2. Install required modules:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Object Registration 

**First of all, open register.py**

1. **Load 3d object parametres.**
   
You need to present object's corners points for model. object's corners format is a numpy array of 4 point, each points is measured in real metres with float32 type (x, y, z).

Start with top-left corner point, and choose default values (0, 0, 0).
Next comes top-right corner that is (x, 0, 0)
Then botton-right point with (x, y, 0).
The last is bottom-left point (0, y, 0).
*Where x, y - width and lenght of the object.

Here is an example:

```python
    object_corners_3d = np.array([
        [0, 0, 0],  # Top-left
        [0.14, 0, 0],  # Top-right
        [0.14, 0.21, 0],  # Bottom-right
        [0, 0.21, 0],  # Bottom-left

    ], dtype="float32") # example of object_corners_3d
```

2. **Register object** using register_to_model function

This function register object on given image. Function takes 6 params:
-   :param object_corners_3d: np.ndarray, 3d coordinate points of object, dtype == np.float32
-   :param input_image: str, path to original image of object
-   :param output_image: str, path to where debug image should be saved
-   :param crop_method: str, the method of cropping, 'corner' (don't crop) or 'manual' (place points directly on image)
-   :param feature_method: str, the method of registration, base - SIFT
-   :param register_output: str, the path to where registration parameters should be saved in .npz format

Example:

```python
    register_to_model(object_corners_3d, os.path.join(MAIN_DIR, "ExampleFiles\\new_book_check\\book_3.jpg"),
                      os.path.join(MAIN_DIR, "ExampleFiles\\OutputFiles\\OutputImages\\output_script_test.jpg"),
                      os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_test.npz"), 'corner', "SIFT")
```
3. **About cropping the photo**
   
If the object is fully presented on the photo, takes the majority of the photo place and there is nothing that could identify the object wrong, then use 'corner' crop method
Othewise if you want to crop the photo, use 'manual' crop method. 
First, ou will see an example of how to crop the image.

![corners_choice_example](https://github.com/user-attachments/assets/60989e46-4b3a-4bca-9560-6ca40693047f)

Then you will see your photo. You have to choose object corners by clicking on them with your left mouse button in a certain order:
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

### Object detection

After completed registration you can detect object on image or photo

1. **Set detector with model parameters.** using set_detector function.

This function set detector using model params. Has 3 input params and 1 output:
-    :param model_params_file: str, path to where model should be saved
-    :param camera_params_file: str, path to saved Camera parameters
-    :param use_flann: bool, use FLANN-based matching or not
-    :return: Detector, object of Detector class

Example:

```python
    detector = set_detector(os.path.join(MAIN_DIR, "ExampleFiles\\ModelParams\\model_test.npz"),
                            os.path.join(MAIN_DIR, "ExampleFiles\\CameraParams\\CameraParams.npz"), True)
```

2. **Photo detection** using detect_photo function.

This function detects object on the photo. Has 3 input params:
-   :param detector: Detector, object of Detector class
-   :param input_file: str, path to image where we use detector
-   :param output_file: str, path to store the image with detected photo

Example:

```python
    detect_photo(detector, os.path.join(MAIN_DIR, "ExampleFiles\\examples\\images\\new_book_check.png"),
                               os.path.join(MAIN_DIR, "ExampleFiles\\OutputFiles\\OutputImages\\contours_drawn.png"))
```

3. **Video detection** using track_frame function.

This func tracks object on video (with detection every track_length) and save video to output. Has 6 parameters:
-   :param detector: Detector, detector, that used to detect object
-   :param video_path: str, path to original video
-   :param output_path: str, destination where video should be saved
-   :param track_length: int, length of frames that should be tracked until new detection
-   :param fps: int, frame per second (optional), 30 fps is usually used
-   :param color: tuple, color of frame in BGR color scheme (255, 0 , 0) - Blue

Example:

```python
    track_frame(detector, os.path.join(MAIN_DIR, "ExampleFiles\\new_book_check\\new_book_video_main.mp4"),
                os.path.join(MAIN_DIR,
                             "ExampleFiles\\OutputFiles/OutputVideos\\new_book_video_main_result_new_color.mp4"),
                60, 30, (0, 0, 255))
```

## Scripts
### Running the Script with Command-Line Arguments DO NOT WORK RIGHT NOW(
You can also run the script directly from the command line using argparse to specify the parameters. This allows you to register images or detect objects via terminal commands.

Examples presented below.

1. **Register an Image**

In that case, in register.py after

```python
if __name__ == "__main__":
```

must be only: 

```python
parse_args_and_execute()
```

An example of how to run a command to register an image:

```
python src/register.py register --input_image "ExampleFiles/new_book_check/book_3.jpg" --output_image "ExampleFiles/OutputFiles/OutputImages/output_script_test.jpg" --crop_method "corner" --points 0 0 0 0.14 0 0 0.14 0.21 0 0 0.21 0 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_test.npz"

```

2. **Detect Features in an Image or Video**

In that case, in detect.py after

```python
if __name__ == "__main__":
```

must be only: 

```python
parse_args_and_execute()
```

An example of how to run a command to detect features:

```
    python detect.py detect --model_input "../ModelParams/model_test.npz" --camera_params "../CameraParams/CameraParams.npz" --input_video "../new_book_check/new_book_video_main.mp4" --video --output_video "../OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4"
```

3. **Arguments:**

   **Register subcommand:**
   
- --input_image: Path to input image for registration.
- --output_image: Path to save the registered image.
- --crop_method: Method for cropping the image (options: 'manual', 'corner').
- --points: List of 3D points (x1 y1 z1 x2 y2 z2 ... xn yn zn). Default is 4 points or 12 coordinates.
- --feature_method: Feature detection method (ORB, KAZE, AKAZE, BRISK, SIFT).
- --model_output: Path to save the model parameters.

   **Detect subcommand:**
   
- --model_input: Path to the saved model file.
- --camera_params: Path to the camera parameters file for detection.
- --input_image: Path to the input image for detection.
- --input_video: Path to the input video for detection.
- --video: Flag if you want to detect video, don't use if you want to detect photo.
- --output_image: Path to output image (if u detected photo) after detection.
- --output_video: Path to output video (if u detected video) after detection.

## License

Not ready yet
