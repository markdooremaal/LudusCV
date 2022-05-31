from NeuralNetwork import NeuralNetwork
import cv2

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

def main():
    nn = NeuralNetwork("../models/30052022-start-pos.onnx", ["startingpos"])

    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()

        class_ids, confidences, boxes = nn.detect_on_frame(frame, True)

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20),
                          (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, "{0} - {1:.2f}".format(nn.class_list[classid], confidences[int(confidence)]), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break


if __name__ == '__main__':
    main()
