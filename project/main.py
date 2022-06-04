from YoloV5 import YoloV5
from PoseDetector import PoseDetector
from Gui import Gui
import cv2

def main():

    # Load all the classes
    nn = YoloV5("../models/24052022.onnx", ["paddle", "light"])
    pose_detect = PoseDetector()
    gui = Gui()

    # Capture
    # capture = cv2.VideoCapture("../experiments/example_data/Camera 1.mov")
    capture = cv2.VideoCapture(0)

    user_started = False
    counter = 50

    while True:
        _, frame = capture.read()

        # Check for the paddle and lights
        nn.draw_on_frame(frame)
        paddle_detected = nn.is_in_frame(frame, "paddle")
        lights_detected = nn.is_in_frame(frame, "light")

        # Detect the starting position
        # frame = pose_detect.draw_pose(frame)
        # frame = pose_detect.draw_labeled_landmarks(frame)
        valid = pose_detect.is_valid_position(frame)
        counter = counter + 1 if not valid else 0

        in_starting_pos = False
        hit_started = False

        if valid or counter < 3: # At least three frames need to detect that the person has moved away from the starting postion to prevent false positives
            in_starting_pos = True
            user_started = True
        elif user_started and counter > 50:
            user_started = False
        elif user_started:
            hit_started = True

        # Draw the GUI
        frame = gui.draw(frame, paddle_detected=paddle_detected, lights_detected=lights_detected, in_starting_position=in_starting_pos, hit_started=hit_started)

        # Output
        cv2.imshow("output", frame)
        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
