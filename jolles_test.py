import numpy as np
import cv2

# Selecteert de webcam
cap = cv2.VideoCapture(0)

# Neemt continue foto's ('frames') aka video
while True:
    ret, frame = cap.read()

    # Minimale en maximale *RGB* waardes
    # 163 255 206
    # 76 255 206
    # TODO: Op internet wordt aangeraden om HSV te gebruiken omdat deze minder afhankelijk is van hoe de lichtinval is
    MIN_RGB = np.flip(np.array([80, 230, 180], dtype=np.uint8))
    MAX_RGB = np.flip(np.array([200, 255, 255], dtype=np.uint8))

    # Selecteer alle pixels die binnen de MIN en MAX vallen
    mask = cv2.inRange(frame, MIN_RGB, MAX_RGB)

    # 'Omcirkelen' van alle vlakken die binnen de mask vallen
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Toont wat er gevonden is
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 10)

    # Check of er iets is gevonden.
    if len(contours) > 0:

        # Pak het grootste vlak
        c = max(contours, key=cv2.contourArea)
        # En teken er een rechtboek omheen
        x, y, w, h = cv2.boundingRect(c)

        # Plaats dit rechthoek op de video
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 50)

        print("JAA het lampje is gezien!")

    # Draait het beeld om
    mirrored = cv2.flip(frame, 1)

    # Toont de mask (zwart-wit)
    # cv2.imshow('Mask', mask)


    # Toont de processed video
    cv2.imshow('Output', mirrored)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()