import cv2
import datetime

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _, frame = cap.read()
    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x, y, w, h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        
        