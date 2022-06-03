GUI_HEIGHT = 35
GUI_LENGTH = 500

def drawGUI(cv2, frame, paddleDetected, lightsDetected, inStartingPosition):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    cv2.rectangle(frame, (0, frameHeight), (frameWidth, frameHeight - GUI_HEIGHT), (255, 255, 255), -1)
    cv2.putText(frame, "Paddle detected:", (0, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, "Lights detected:", (190, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, "In starting position:", (380, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if(paddleDetected):
      cv2.rectangle(frame, (135, frameHeight), (175, frameHeight - GUI_HEIGHT), (0, 255, 0), -1)
      cv2.putText(frame, "Yes", (140, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
      cv2.rectangle(frame, (135, frameHeight), (175, frameHeight - GUI_HEIGHT), (0, 0, 255), -1)
      cv2.putText(frame, "No", (140, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if (lightsDetected):
      cv2.rectangle(frame, (320, frameHeight), (360, frameHeight - GUI_HEIGHT), (0, 255, 0), -1)
      cv2.putText(frame, "Yes", (325, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
      cv2.rectangle(frame, (320, frameHeight), (360, frameHeight - GUI_HEIGHT), (0, 0, 255), -1)
      cv2.putText(frame, "No", (325, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if (inStartingPosition):
      cv2.rectangle(frame, (540, frameHeight), (580, frameHeight - GUI_HEIGHT), (0, 255, 0), -1)
      cv2.putText(frame, "Yes", (545, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
      cv2.rectangle(frame, (540, frameHeight), (580, frameHeight - GUI_HEIGHT), (0, 0, 255), -1)
      cv2.putText(frame, "No", (545, frameHeight - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
