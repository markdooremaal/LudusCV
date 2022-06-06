# from YoloV5 import YoloV5
# from PoseDetector import PoseDetector
# from Gui import Gui
import cv2
import numpy as np
from DetectPaddle import DetectPaddle
from StartingPose import StartingPose
from GuiOverlay import GuiOverlay


class Application:
    parsers = None
    output = None
    frame = None

    def __init__(self):
        """
        First, let's construct the application object.
        """
        self.parsers = {
            "yolo": DetectPaddle(self),
            "starting_pose": StartingPose(self),
            "gui": GuiOverlay(self),
        }

    def run(self):
        """
        Will keep grabbing frames and run all the parser on it.
        :return:
        """
        capture = self.get_capture()
        while True:

            # Read frame
            _, self.frame = capture.read()
            self.output = np.copy(self.frame)

            # Parse Frame
            for parser in self.parsers:
                self.parsers[parser].parse_frame(self.output)

            # Show frame
            cv2.imshow("output", self.output)

            if cv2.waitKey(1) > -1:
                break

    def get_capture(self):
        """
        Will return the input stream
        :return:
        """
        return cv2.VideoCapture(0)


# def main():
#     # Load all the classes
#     nn = YoloV5("../models/24052022.onnx", ["paddle", "light"])
#     pose_detect = PoseDetector()
#     gui = Gui()
#
#     # Capture
#     capture = cv2.VideoCapture("../experiments/example_data/Camera 1.mov")
#     # capture = cv2.VideoCapture(0)
#
#     user_started = False
#     counter = 50
#
#     while True:
#         _, frame = capture.read()
#
#         # Check for the paddle and lights
#         nn.draw_on_frame(frame)
#         paddle_detected = nn.is_in_frame(frame, "paddle")
#         lights_detected = nn.is_in_frame(frame, "light")
#
#         # Detect the starting position
#         # frame = pose_detect.draw_pose(frame)
#         # frame = pose_detect.draw_labeled_landmarks(frame)
#         valid = pose_detect.is_valid_position(frame)
#         counter = counter + 1 if not valid else 0
#
#         in_starting_pos = False
#         hit_started = False
#
#         if valid or counter < 3:  # At least three frames need to detect that the person has moved away from the starting postion to prevent false positives
#             in_starting_pos = True
#             user_started = True
#         elif user_started and counter > 50:
#             user_started = False
#         elif user_started:
#             hit_started = True
#
#         # Draw the GUI
#         frame = gui.draw(frame, paddle_detected=paddle_detected, lights_detected=lights_detected,
#                          in_starting_position=in_starting_pos, hit_started=hit_started)
#
#         # Output
#         cv2.imshow("output", frame)
#         if cv2.waitKey(1) > -1:
#             print("finished by user")
#             break


if __name__ == '__main__':
    app = Application()
    app.run()
    # main()
