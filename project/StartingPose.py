from BaseParser import BaseParser
from PoseDetector import PoseDetector


class StartingPose(BaseParser):
    detector = PoseDetector()

    in_starting_pos = False
    hit_started = False

    __user_started = False
    __counter = 0

    def parse_frame(self, output):
        self.detector.draw_pose(output)

        valid = self.detector.is_valid_position(output)
        self.__counter = self.__counter + 1 if not valid else 0

        self.in_starting_pos = False
        self.hit_started = False

        if valid or self.__counter < 1:  # At least three frames need to detect that the person has moved away from the starting postion to prevent false positives
            self.in_starting_pos = True
            self.__user_started = True
        elif self.__user_started and self.__counter > 10:
            self.__user_started = False
        elif self.__user_started:
            self.hit_started = True

