from NeuralNetwork import NeuralNetwork
from ArmDetection import ArmDetection
import cv2

def main():
    nn = NeuralNetwork("../models/24052022.onnx", ["startingpos", "light"])
    arm_detect = ArmDetection()
    capture = cv2.VideoCapture("../experiments/example_data/lampjetest.mp4")

    while True:
        _, frame = capture.read()

        # frame = arm_detect.find_pose(frame)
        lmList, bboxInfo = arm_detect.find_position(frame)
        frame = nn.draw_on_frame(frame)

        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
