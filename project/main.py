import cv2
import numpy as np
from DetectPaddle import DetectPaddle
from StartingPose import StartingPose
from HitParser import HitParser
from GuiOverlay import GuiOverlay
import sys


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

        # If the input is specified in the arguments, use that:
        if len(sys.argv) >= 1:
            try:
                arg = int(sys.argv[1])
            except ValueError:
                arg = sys.argv[1]
            try:
                cap = cv2.VideoCapture(arg)
                return cap
            except:
                print("Cannot parse", arg)

        # Else manually select a webcam
        _, working_ports, _ = self.list_cameras()
        if len(working_ports) > 1:
            camera_id = input("Pick an camera (available: "+str(working_ports)+"): [default="+str(working_ports[0])+"]")
            if not camera_id == "" and working_ports.index(int(camera_id)):
                return cv2.VideoCapture(int(camera_id))
        if len(working_ports) >= 1:
            return cv2.VideoCapture(working_ports[0])
        else:
            print("No camera's available")
            exit()

    def list_cameras(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        """
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
            else:
                is_reading, img = camera.read()
                if is_reading:
                    working_ports.append(dev_port)
                else:
                    available_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports, non_working_ports



if __name__ == '__main__':
    app = Application()
    app.run()
    # main()
