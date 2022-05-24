import numpy as np
import cv2

# Selecteert de webcam
cap = cv2.VideoCapture(0)

# Neemt continue foto's ('frames') aka video
while True:
    _, original_frame = cap.read()

    # Create the HSV frame (Hue, Saturation, Value) and the output frame
    hsv_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
    output_frame = np.copy(original_frame)

    # Minimale en maximale *HSV* waardes
    min_hsv = np.array([65, 30, 211])
    max_hsv = np.array([85, 255, 255])

    # Selecteer alle pixels die binnen de MIN en MAX vallen
    mask = cv2.inRange(hsv_frame, min_hsv, max_hsv)
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)

    # 'Omcirkelen' van alle vlakken die binnen de mask vallen
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Toont wat er gevonden is
    cv2.drawContours(output_frame, contours, -1, (255, 0, 0), 10)

    # Check of er iets is gevonden.
    if len(contours) > 0:

        # Pak het grootste vlak
        c = max(contours, key=cv2.contourArea)
        # En teken er een rechtboek omheen
        x, y, w, h = cv2.boundingRect(c)

        # Plaats dit rechthoek op de video
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 10)

        # print("JAA het lampje is gezien!")

    # Draait het beeld om
    mirrored = cv2.flip(output_frame, 1)

    result_masked = cv2.bitwise_and(original_frame, original_frame, mask=mask)

    # Toont de mask
    cv2.imshow('Mask', result_masked)

    # Toont de processed video
    cv2.imshow('Output', mirrored)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()