# PeyeTracking Python package and Python Dash visualization platform

## Brief project introduction
The interpretation of the same gaze data could be pretty different if researchers use different method to identify the fixations. The off-the-shelf algorithms in eye tracker now like Tobii 4c are mainly speed-based-method, which means that identifying fixationsonly based on eyeball movement speed. This could make it easy to ignore the dependence of fixation distribution on the methods the researchers use. For study using fixation statistics, multiple parameter settings can help with
getting more complete picture of the fixation distribution

In this project, I created:
* PeyeTracking Package: A python package enables more efficient and more customized fixation identification.
* Python Dash visualization platform: A dashboard developed under Python Dash with the following functionalities:
  * Preprocessing raw gaze data with PeyeTracking package
  * Fixation identification with PeyeTracking package
  * Eye gaze speed visualization with identified fixations
  * Fixation annotation on experiment screen recording video
  * Experiment screen recording video frame inspection with identified fixations

## Install PeyeTracking package locally

```
$cd capstone_shiyao
$pip install -e PeyeTracking
```

In this way, you can also simply make changes to the package when you need.

### Import functions in PeyeTracking
```
from PeyeTracking.data_preprocess import pre_process
from PeyeTracking.fixation_classification import fixation_detection, visualize_fixation, get_speed
```
## Launch the dashboard
### Requirements for raw gaze data and screen recording video:
* This version only supports raw gaze data from Tobii eye tracker, here is an [example](https://github.com/shiyaol/capstone_shiyao/blob/main/example_data)
* For the screen recording videoe, please transfer the frame rate to **25** first, you can use the package [FFmpeg](https://www.ffmpeg.org/), below is an example converting movie_13.mp4 to 13_movie.mp4 with a frame rate of 25:

```
$pip install ffmpeg
$cd <video folder>
$ffmpeg -i movie_13.mp4 -r 25 -y 13_movie.mp4
```

### Run the dashboard on local server
* In IDE: Simply run [app.py](https://github.com/shiyaol/capstone_shiyao/blob/main/gaze_app/app.py), the dashboard will be lauched on the local server address(http://127.0.0.1:8050/).

### The frame and video folder:
* The [frames](https://github.com/shiyaol/capstone_shiyao/tree/main/gaze_app/frames) folder will store your generated frames. Everytime you rerun the dashboard, images under this folder will be deleted automatically.
* Please put the screen recording video under the [gaze_app](https://github.com/shiyaol/capstone_shiyao/tree/main/gaze_app) folder, the generated videos will also be saved in this folder.
