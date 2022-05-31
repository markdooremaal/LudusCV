from YoloV5 import YoloV5
from PoseDetector import PoseDetector
import cv2

def main():
    # nn = YoloV5("../models/24052022.onnx", ["startingpos", "light"])
    pose_detect = PoseDetector()
    # capture = cv2.VideoCapture("../experiments/example_data/lampjetest.mp4")
    capture = cv2.VideoCapture(2)

    while True:
        _, frame = capture.read()

        frame = pose_detect.draw_pose(frame)
        # frame = pose_detect.draw_landmarks(frame)
        valid = pose_detect.is_valid_position(frame)
        cv2.putText(frame, "Correct", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) if valid else cv2.putText(frame, "Incorrect", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
