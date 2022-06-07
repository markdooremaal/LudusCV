# LudusCV
An OpenCV POC for the Ludus Paddle

## Required packages
* opencv-python
* cvzone
* mediapipe (or mediapipe-silicon for Apple Silicon Macs)
* pandas
* numpy 

## Installation instructions
We recommend installing all the required packages with pip or pipenv.
*Note* that when using pipenv mediapipe needs to be installed seperately because of the different versions depending on the architecture you are using.

## How to run
Navigate to `/project/` and enter the following command in the commandline:
`python3 main.py`.

To close the program press `esc` or `ctrl+c` in the terminal.

## Change the camera
By default the program will use your webcam, to change to another camera change change the `0` within `return cv2.VideoCapture(0)` to a desired index.
