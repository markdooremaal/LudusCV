from BaseParser import BaseParser
from PoseDetector import PoseDetector


class StartingPose(BaseParser):
    detector = PoseDetector()

    def parse_frame(self, output):
        self.detector.draw_pose(output)

