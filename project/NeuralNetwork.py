import cv2
import numpy as np

class NeuralNetwork:
    def __init__(self, path_to_onnx, class_list=None, sizes = (416, 416)):
        self.class_list = class_list
        self.class_list = class_list if class_list is not None else []

        network = cv2.dnn.readNet(path_to_onnx)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.network = network

        self.sizes = sizes

    class_list = None
    network = None
    sizes = (416, 416)

    # Will run the predictions
    def detect(self, image, net):
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, self.sizes, swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds

    # Will surround the detected objects
    def wrap_detection(self, input_image, output_data, low_confidence=False):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        score_threshold = .001 if low_confidence else 0.25
        nms_threshold = .005 if low_confidence else 0.45

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.sizes[0]
        y_factor = image_height / self.sizes[1]

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= score_threshold or low_confidence:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > score_threshold or low_confidence):
                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(
                    ), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    # Will convert a frame into an array that yolo has been trained on.
    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    # Use this function to detect on an image:
    def detect_on_frame(self, frame, low_confidence = False):
        input_image = self.format_yolov5(frame)
        outs = self.detect(input_image, self.network)
        return self.wrap_detection(input_image, outs[0], low_confidence)
