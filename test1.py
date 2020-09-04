import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('model/face_detect_cascade.xml')
image = cv2.imread('Data/train/Alejandro_Toledo/Alejandro_Toledo_0001.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    face = image[y:y+h,x:x+w,:].copy()
    face = cv2.resize(face,(160,160))
    cv2.imshow('1',face)
cv2.waitKey(0)