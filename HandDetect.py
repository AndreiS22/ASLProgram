import cv2
import numpy as np 

hand_detect = cv2.CascadeClassifier('data/cascade.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hands = hand_detect.detectMultiScale(gray, 4, 7)
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 20, 40), 2)
    cv2.imshow("KBOOOM!!", img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destrotAllWindows()