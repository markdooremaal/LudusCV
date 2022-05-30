import cv2
import numpy as np

INPUT_WIDTH = 416
INPUT_HEIGHT = 416
class_list = ['Paddle', 'Lights']

# Will load the model into OpenCV DNN
def build_model():
    net = cv2.dnn.readNet("../models/24052022.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# Will run the predictions
def detect(image, net):
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

# Will surround the detected objects
def wrap_detection(input_image, output_data, low_confidence = False):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    score_threshold = .001 if low_confidence else 0.25
    nms_threshold = .005 if low_confidence else 0.45

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

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
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


# START!
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
net = build_model()
capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture("example_data/lampjetest.mp4")

while True:
    _, frame = capture.read()
    output = np.copy(frame)

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):

        if classid == 0:
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(output, box, color, 2)
            cv2.rectangle(output, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(output, "{0} - {1:.2f}".format(class_list[classid], confidences[int(confidence)]), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

            (x, y, w, h) = box
            extract = np.copy(frame[y-10:y + h+10, x-10:x + w+10])
            inputImage = format_yolov5(extract)
            outs = detect(inputImage, net)

            class_ids, confidences, boxes = wrap_detection(inputImage, outs[0], True)

            for (classid, confidence, box) in zip(class_ids, confidences, boxes):

                if classid == 1:
                    color = colors[int(classid) % len(colors)]
                    cv2.rectangle(extract, box, color, 2)
                    cv2.rectangle(extract, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                    cv2.putText(extract, "{0} - {1:.2f}".format(class_list[classid], confidences[int(confidence)]),
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

            if extract.size != 0:
                extractOutput = cv2.resize(extract, dsize=(500, 500))
                cv2.imshow("extract", extractOutput)

    cv2.imshow("output", output)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break
