from BaseParser import BaseParser
import time
import cv2


class HitParser(BaseParser):
    __can_hit = -100
    __user_has_hit = -100
    __last_starting_pos = -100
    __first_light_detected = -100

    __hit_time_start = -1
    hit_time = -1

    can_hit = False
    user_has_hit = False
    user_in_starting_pos = False
    paddle_hit = False
    to_early = False

    def parse_frame(self, output):

        # Starting pos
        if self.application.parsers["starting_pose"].in_starting_pos:
            self.__last_starting_pos = self.application.frame_index
            self.user_in_starting_pos = True
        elif self.__last_starting_pos < self.application.frame_index - 2:
            self.user_in_starting_pos = False

        # Light detection
        if self.application.parsers["yolo"].lights_detected:
            self.__can_hit = self.application.frame_index
        self.can_hit = self.__can_hit > self.application.frame_index - 5
        if self.can_hit and self.__first_light_detected < self.application.frame_index - 20:
            self.__first_light_detected = self.application.frame_index
            self.__hit_time_start = round(time.time() * 1000)
            self.hit_time = -1

        # Check if the hit has been started
        if not self.user_in_starting_pos:
            self.__user_has_hit = self.application.frame_index
        self.user_has_hit = not self.user_in_starting_pos and self.application.frame_index - 2 < self.__user_has_hit and self.__last_starting_pos > self.application.frame_index - 10

        # Check if the paddle has been hit
        self.paddle_hit = self.is_paddle_hit()
        if self.paddle_hit:
            if self.hit_time == -1 and self.__hit_time_start > 0:
                self.hit_time = round(time.time() * 1000) - self.__hit_time_start
            box = self.application.parsers["yolo"].paddle_box
            cv2.rectangle(output, box, (0, 255, 255), 5)

        # Check if the hit was too early
        self.to_early = self.hit_is_to_early()

    def is_paddle_hit(self):
        if self.application.parsers["yolo"].paddle_box is not None and self.application.parsers["starting_pose"].detector.last_position:
            (x, y, w, h) = self.application.parsers["yolo"].paddle_box
            pos = self.application.parsers["starting_pose"].detector.last_position
            lpos = pos['left_thumb']
            rpos = pos['right_thumb']
            return x < lpos['x'] < x + w and y < lpos['y'] < y + y or x < rpos['x'] < x + w and y < rpos['y'] < y + y
        return False

    def hit_is_to_early(self):
        return self.user_has_hit and not self.can_hit

