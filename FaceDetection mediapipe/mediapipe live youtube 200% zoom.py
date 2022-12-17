import cv2
import mss
import numpy as np
import mediapipe as mp
import time
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
prevTime = 0

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.7) as face_detection:
  while True:
    with mss.mss() as sct:
      monitor = {'top': 226, 'left':312, 'width': 1280, 'height': 720}
      img = np.array(sct.grab(monitor))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      #cv2.imshow("img", img)
  

    #Convert the BGR image to RGB.
    image = cv2.cvtColor( img, cv2.COLOR_RGB2BGR)
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor( img, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
        
    
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 196, 255), 1)
    cv2.imshow('BlazeFace Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

