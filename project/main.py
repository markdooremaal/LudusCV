from YoloV5 import YoloV5
from PoseDetector import PoseDetector
import cv2

def main():
    # nn = YoloV5("../models/24052022.onnx", ["startingpos", "light"])
    pose_detect = PoseDetector()
    # capture = cv2.VideoCapture("../experiments/example_data/Camera 1.mov")
    capture = cv2.VideoCapture(0)

    user_started = False
    counter = 50

    while True:
        _, frame = capture.read()

        # frame = pose_detect.draw_pose(frame)
        # frame = pose_detect.draw_labeled_landmarks(frame)
        valid = pose_detect.is_valid_position(frame)
        counter = counter + 1 if not valid else 0

        if valid or counter < 3:
            color = (0, 255, 0)
            text = "In starting position"
            user_started = True
            frame[:, :, 2] = frame[:, :, 2] / 255 * 200
            frame[:, :, 0] = frame[:, :, 0] / 255 * 200
        else:
            color = (0, 0, 255)
            if user_started:
                text = "Started with hit!"
                if counter > 50:
                    user_started = False
            else:
                text = "Go to starting position"
        cv2.putText(frame, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        if user_started:
            cv2.putText(frame, "Frames since hit start: " + str(counter), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
