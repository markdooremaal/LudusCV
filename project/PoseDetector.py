import cv2
from cvzone.PoseModule import PoseDetector as Detector


class PoseDetector:
    detector = None

    def __init__(self):
        self.detector = Detector()

    def __find_pose(self, frame, draw=True):
        return self.detector.findPose(frame, draw)

    def draw_pose(self, frame):
        return self.__find_pose(frame, True)

    def draw_landmarks(self, frame):
        landmarks, bounding_box = self.find_position(frame)
        if len(landmarks) > 1:
            for landmark in range(0, len(landmarks)):
                cv2.putText(frame, str(landmark), (landmarks[landmark][1], landmarks[landmark][2]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def draw_labeled_landmarks(self, frame):
        landmarks, bounding_box = self.find_position(frame)

        if len(landmarks) == 0:
            return frame

        positions = self.__get_position(landmarks)
        for position in positions:
            cv2.putText(frame, position, (positions[position]["x"], positions[position]["y"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, "center", bounding_box['center'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame

    def find_position(self, frame):
        # See https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
        return self.detector.findPosition(self.__find_pose(frame, False), bboxWithHands=False, draw=False)

    def is_valid_position(self, frame):
        landmarks, bounding_box = self.find_position(frame)

        if len(landmarks) == 0:
            return False

        position = self.__get_position(landmarks)
        height, width, _ = frame.shape
        error_margin = round(width / 100)
        shoulder_distance = abs(position["right_shoulder"]["x"] - position["left_shoulder"]["x"])
        max_thumb_to_nose = round(shoulder_distance / 2) + error_margin
        max_offset = round(shoulder_distance / 3) + error_margin

        checks = [
            position["right_elbow"]["y"] < height,
            position["left_elbow"]["y"] < height,
            position["right_elbow"]["y"] - max_offset < position["right_shoulder"]["y"],
            position["left_elbow"]["y"] - max_offset < position["left_shoulder"]["y"],
            position["right_wrist"]["y"] - max_offset < position["right_elbow"]["y"],
            position["left_wrist"]["y"] - max_offset < position["left_elbow"]["y"],
            abs(position["left_thumb"]["x"] - position["nose"]["x"]) < max_thumb_to_nose,
            abs(position["left_thumb"]["y"] - position["nose"]["y"]) < max_thumb_to_nose,
            abs(position["right_thumb"]["x"] - position["nose"]["x"]) < max_thumb_to_nose,
            abs(position["right_thumb"]["y"] - position["nose"]["y"]) < max_thumb_to_nose,
        ]

        return all(checks)

    def __get_x_y_z(self, landmark):
        return {
            "x": landmark[1],
            "y": landmark[2],
            "z": landmark[3],
        }

    def get_position(self, landmarks):
        return self.__get_position(landmarks)

    def __get_position(self, landmarks):
        return {
            "left_shoulder": self.__get_x_y_z(landmarks[11]),
            "right_shoulder": self.__get_x_y_z(landmarks[12]),
            "left_elbow": self.__get_x_y_z(landmarks[13]),
            "right_elbow": self.__get_x_y_z(landmarks[14]),
            "left_wrist": self.__get_x_y_z(landmarks[15]),
            "right_wrist": self.__get_x_y_z(landmarks[16]),
            "left_thumb": self.__get_x_y_z(landmarks[21]),
            "right_thumb": self.__get_x_y_z(landmarks[22]),
            "nose": self.__get_x_y_z(landmarks[0]),
            "left_hip": self.__get_x_y_z(landmarks[23]),
            "right_hip": self.__get_x_y_z(landmarks[24]),
            "left_heel": self.__get_x_y_z(landmarks[29]),
            "right_heel": self.__get_x_y_z(landmarks[30]),
        }
