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
## Launch the Dashboard
### Requirements for raw gaze data and screen recording video:
* This version only supports raw gaze data from Tobii eye tracker, here is an example:

### Run the dashboard on local server
* In IDE: Simply run [app.py](https://github.com/shiyaol/capstone_shiyao/blob/main/gaze_app/app.py), the local server address(http://127.0.0.1:8050/) will show in the output.

