# Аugmented Reality
# Object registration and detection using OpenCV

This project provides tools for registrating objects (images), detecting in images and videos and rendering modelss over detected objects. 

## Table of Contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

## Description

This project enables:
- Loading camera calibration parameters.
- Loading image, cropping it (through clicking on the object corners on the picture) and loading 3d parameters of the object .
- Registrating object from image or webcamera for futher detection.
- Loading camera calibration parameters.
- Detecting an object in images and videos from prepared files or online using webcamera with algorithms: "ORB", "KAZE", "AKAZE", "BRISK", "SIFT".
- Saving detection results as image or video.
- Running registration and detection scripts
- Rendering modelss over detected objects

## Requirements

Major ones:

- matplotlib==3.9.2
- numpy==2.1.3
- opencv-python==4.10.0.84
- trimesh==4.6.1
- pyrender==0.1.45

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

The program is launched by scripts using command line parser.

First, you have to register you object and get a model that is used to detect your object. When finished registration, you can run detection script.

### Object Registration 

The script for registration looks like:

```bash
python register.py --param_name1 param_value1 --param_name2 param_value2 --param_name3 param_value3 ...
```

#### Registration has several params, some of them optional, some of them are required:

- input_image, type=str, required=False, "Path to input image for registration"
  
- output_image, type=str, required=True, "Path to save the registered image"
  
- w, type=float, required=True, "Width of the object"

- h, type=float, required=True, "Height of the object"

- feature_method, type=str, choices=["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"], "Feature detection method (default='ORB')"

- model_output, type=str, required=True, "Path to save the model parameters"

- webcam, action='store_true', "if you want to register an object using your webcam"

If you want to register an object from the file, use 'input_image'. If you want to register an object from your webcamera, use 'webcam'.

After running the script, you wil have to crop the image (more details below).

#### Examples of registration scipts with Command-Line Arguments:

```bash
python register.py --input_image "ExampleFiles/new_book_check/book_3.jpg" --output_image "ExampleFiles/OutputFiles/OutputImages/output_script_test.jpg" --w 0.14 --h 0.21 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_test.npz"
```

```bash
python register.py --output_image "exampleFiles/OutputFiles/OutputImages/output_varior_book.png" --w 0.13 --h 0.205 --feature_method "SIFT" --model_output "ExampleFiles/ModelParams/model_varior_book.npz" --webcam
```


#### About cropping the photo

User have to identify object corners for futher work. It is done via cropping the image by object's corners.

First, you will see an example of how to crop the image.

![corners_choice_example](https://github.com/user-attachments/assets/60989e46-4b3a-4bca-9560-6ca40693047f)

Then you will see your photo. You have to choose object corners by clicking on them with your left mouse button in a certain order:
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Then press 'enter'.

### Object detection

When finished registration and you got a model of your object, you can run detection script.

The script for detection looks like:

```bash
python detect.py --param_name1 param_value1 --param_name2 param_value2 --param_name3 param_value3 ...
```

#### Detection has several params, some of them optional, some of them are required:

- model, type=str, help="Path to the saved model file"

- demo, action='store_true', "Use the default object model from the repository"

- camera_params, type=str, required=False, "Path to camera parameters file"

- input, type=str, required=False, "Path to input image or video for detection"

- video, action='store_true', "if you want to detect video, don't use if you want to detect photo"

- output, type=str, required=True, "Path to output image or video after detection"

- use_tracker, action='store_true', "Use if you want to use tracking"

- web_camera, action='store_true', "Use if you want to use your camera"

- visualize_matches, action='store_true', "Use if you want to visualize matches with the reference image"

If you want to detect an object from your model, use 'model'. If you want to test detection with the model from repo use 'demo'.

If you want to detect an object from the file, use 'input'. If you want to detect an object from your webcamera, use 'web_camera'.

If you know your camera parametres, you can use 'camera_params'. Otherwise, the program will use default camera parametres.

If you want to detect an object on video, use 'video', and all the input and output file should be 'mp4' format. So, you want ti detect an object on photo, all the input and output file should be photo format.

**After running the script with your webcamera:**

- first, place your object, so your object will be visible for the camera

- second, press 'q' to start the detection of your object

- finally, press 'q' to end detection.

#### Examples of detection scipts with Command-Line Arguments:

```bash
python detect.py --model "ExampleFiles/ModelParams/model_test.npz" --camera_params "ExampleFiles/CameraParams/CameraParams.npz" --input "ExampleFiles/new_book_check/new_book_video_main.mp4" --video --output "ExampleFiles/OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4"
```

```bash
python detect.py --demo --camera_params "ExampleFiles/CameraParams/CameraParams.npz" --input "ExampleFiles/new_book_check/new_book_video_main.mp4" --video --output "ExampleFiles/OutputFiles/OutputVideos/new_book_video_main_result_new_color.mp4" --use_tracker
```

```bash
python detect.py --model "ExampleFiles/ModelParams/model_varior_book_iphone.npz" --video --output "ExampleFiles/OutputFiles/OutputVideos/varior_book_result_iphone.mp4" --use_tracker --web_camera 
```

## License

Not ready yet
