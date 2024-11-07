# Object registration and detection using OpenCV

This project provides tools for registrating and detecting objects (images) in images and videos. 

## Table of Contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

## Description

This project enables:
- Loading camera calibration parameters.
- Loading image, cropping it and giving the volume parameter for 3d frame.
- Registrating object from image for futher detection.
- Detecting an object in images and videos using algorithms: "ORB", "KAZE", "AKAZE", "BRISK", "SIFT".
- Displaying and saving detection results as image or video.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)

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

### Object Registration and Detection

1. **Create Model class object and load camera parameters** using Model() and the `load_camera_params` method.
2. **Load an image and register the object** using the `upload_image` and 'register' (with a selected method: "ORB", "KAZE", "AKAZE", "BRISK", "SIFT") methods respectively
3. **Crop the image if needed** using the `crop_image_by_clicks' or 'crop_image_by_points' method
4. **Save the model parameters if needed** using the `save_to_npz' method
5. **Load model parameters if needed** using the `load' method
6. **Create Detector class object and load  Model parametres** using Detector() and the 'get_model_params' or 'load_model_params' method; then instance_method is needed
7. **Detect object (image) on image/photo** using the `detect_image' or 'detect_video' method

Example:

```python
from registration import Model
from detection import Detector

# Step 1: Create Model class object and load camera parameters
model = Model()
model.load_camera_params("./CameraParams/CameraParams.npz")

# Step 2: Load an image and register the object
model.upload_image('path/to/input_image.jpg', 'output_image.jpg')
model.register('SIFT') # "ORB", "KAZE", "AKAZE", "BRISK", "SIFT"

# Step 3: Crop the image if needed
model.crop_image_by_clicks() #or
# points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
# model.crop_image_by_points(points)

# Step 4: Save the model parameters if needed
model_path = 'model_params.npz'
model.save_to_npz(model_path)

# Step 5: Load model parameters if needed
model = Model.load(model_path)

# Step 6: Create Detector class object and load  Model parametres
detector = Detector()
detector.get_model_params(model) # or
# detector.load_model_params(model_path)
detector.instance_method(True)

# Step 7: Detect object (image) on image/photo
detector.detect_image('path/to/target_image.jpg', useFlann=True, drawMatch=True) #or
#detector.detect_video('path/to/target_video.mp4'")
```

## License

Not ready yet
