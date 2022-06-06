from BaseParser import BaseParser
from YoloV5 import YoloV5


class DetectPaddle(BaseParser):

    yolo = None

    def __init__(self, application):
        self.yolo = YoloV5("../models/24052022.onnx", ["paddle", "light"])
        super().__init__(application)

    def parse_frame(self, output):
        self.yolo.draw_on_frame(output)
