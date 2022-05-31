from YoloV5 import YoloV5
from PoseDetector import PoseDetector
import cv2

def main():
    # nn = YoloV5("../models/24052022.onnx", ["startingpos", "light"])
    pose_detect = PoseDetector()
    # capture = cv2.VideoCapture("../experiments/example_data/lampjetest.mp4")
    capture = cv2.VideoCapture(0)

    user_started = False
    counter = 0

    while True:
        _, frame = capture.read()

        # frame = pose_detect.draw_pose(frame)
        # frame = pose_detect.draw_labeled_landmarks(frame)
        valid = pose_detect.is_valid_position(frame)
        if valid:
            color = (0, 255, 0)
            text = "In starting position"
            user_started = True
            counter = 0
        else:
            color = (0, 0, 255)
            if user_started:
                counter += 1
                text = "Started with hit!"
                if counter > 50:
                    user_started = False
            else:
                text = "Go to starting position"
        cv2.putText(frame, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)


        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
