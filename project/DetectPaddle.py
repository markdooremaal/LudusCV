from BaseParser import BaseParser
from YoloV5 import YoloV5
import cv2
import numpy as np


class DetectPaddle(BaseParser):
    yolo = None
    paddle_box = None
    lights_detected = False

    __counter = 0

    def __init__(self, application):
        self.yolo = YoloV5("../models/24052022.onnx", ["paddle", "light"])
        super().__init__(application)

    def parse_frame(self, output):

        # Once the light is detected, keep it on for a while
        if self.__counter != 0:
            self.__counter = self.__counter - 1
        elif self.__counter == 0 and self.lights_detected == True:
            self.lights_detected = False

        # Detect the location of the paddle and search for green
        coords = self.yolo.get_coords(output, "paddle")
        if coords is not False:
            (x, y, w, h) = coords
            error_x = round(w / 10)
            error_y = round(h / 10)
            self.paddle_box = [x - error_x, y - error_y, w + error_x * 2, y + error_y]

        if self.paddle_box is not None:
            (x, y, w, h) = self.paddle_box
            extract = output[y:y+h, x:x+w]
            self.light_detected(extract)

        # Render the box around the paddle
        if self.lights_detected:
            cv2.rectangle(output, self.paddle_box, (0, 255, 0), 2)
        elif coords is not None:
            cv2.rectangle(output, self.paddle_box, (200, 200, 255), 2)
        elif self.paddle_box is not None:
            cv2.rectangle(output, self.paddle_box, (0, 0, 255), 2)

    def light_detected(self, segment):
        try:
            hsv_frame = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
        except Exception:
            return

        # Minimale en maximale *HSV* waardes
        min_hsv = np.array([65, 50, 50])
        max_hsv = np.array([85, 255, 255])

        mask = cv2.inRange(hsv_frame, min_hsv, max_hsv)

        green_filled = cv2.countNonZero(mask) / (segment.shape[0] * segment.shape[1]) * 100  # Calculate the % of green
        if green_filled > 0.03:
            self.__counter = 3
            self.lights_detected = True
