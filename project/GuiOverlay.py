from BaseParser import BaseParser
import cv2

class GuiOverlay(BaseParser):
    __gui_height = 35
    __gui_width = 500

    def parse_frame(self, output):

        paddle_detected = self.application.parsers["yolo"].yolo.is_in_frame(output, "paddle")
        lights_detected = self.application.parsers["yolo"].yolo.is_in_frame(output, "light")
        in_starting_position = self.application.parsers["starting_pose"].detector.is_valid_position(output)

        frame_height = output.shape[0]
        frame_width = output.shape[1]

        cv2.rectangle(output, (0, frame_height), (frame_width, frame_height - self.__gui_height), (255, 255, 255), -1)
        cv2.putText(output, "Paddle detected:", (0, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(output, "Lights detected:", (190, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(output, "In starting position:", (380, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                    1)

        if paddle_detected:
            cv2.rectangle(output, (135, frame_height), (175, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(output, "Yes", (140, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(output, (135, frame_height), (175, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(output, "No", (140, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if lights_detected:
            cv2.rectangle(output, (320, frame_height), (360, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(output, "Yes", (325, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(output, (320, frame_height), (360, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(output, "No", (325, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if in_starting_position:
            cv2.rectangle(output, (540, frame_height), (580, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(output, "Yes", (545, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(output, (540, frame_height), (580, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(output, "No", (545, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        