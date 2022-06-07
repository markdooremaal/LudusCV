import cv2
import numpy as np
from DetectPaddle import DetectPaddle
from StartingPose import StartingPose
from HitParser import HitParser
from GuiOverlay import GuiOverlay


class Application:
    parsers = None
    output = None
    frame = None
    frame_index = 0

    def __init__(self):
        """
        First, let's construct the application object.
        """
        self.parsers = {
            "yolo": DetectPaddle(self),
            "starting_pose": StartingPose(self),
            "hit_parser": HitParser(self),
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
            self.frame_index = self.frame_index + 1
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



if __name__ == '__main__':
    app = Application()
    app.run()
    # main()
