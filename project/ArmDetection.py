import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector


class ArmDetection:
    detector = None

    def __init__(self):
        self.detector = PoseDetector()

    def find_pose(self, frame):
        return self.detector.findPose(frame)

    def find_position(self, frame):
        return self.detector.findPosition(self.find_pose(np.copy(frame)), bboxWithHands=False)