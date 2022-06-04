import cv2


class Gui:
    __gui_height = 35
    __gui_width = 500

    def __init__(self, height=35, length=500):
        self.__gui_width = length
        self.__gui_height = height

    def draw(self, frame, paddle_detected, lights_detected, in_starting_position, hit_started):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        cv2.rectangle(frame, (0, frame_height), (frame_width, frame_height - self.__gui_height), (255, 255, 255), -1)
        cv2.putText(frame, "Paddle detected:", (0, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "Lights detected:", (190, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "In starting position:", (380, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if paddle_detected:
            cv2.rectangle(frame, (135, frame_height), (175, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(frame, "Yes", (140, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(frame, (135, frame_height), (175, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(frame, "No", (140, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if lights_detected:
            cv2.rectangle(frame, (320, frame_height), (360, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(frame, "Yes", (325, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(frame, (320, frame_height), (360, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(frame, "No", (325, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if in_starting_position:
            cv2.rectangle(frame, (540, frame_height), (580, frame_height - self.__gui_height), (0, 255, 0), -1)
            cv2.putText(frame, "Yes", (545, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        elif hit_started:
            cv2.rectangle(frame, (540, frame_height), (580, frame_height - self.__gui_height), (255, 0, 255), -1)
            cv2.putText(frame, "HIT", (545, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (540, frame_height), (580, frame_height - self.__gui_height), (0, 0, 255), -1)
            cv2.putText(frame, "No", (545, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
