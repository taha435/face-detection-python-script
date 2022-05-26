import cv2
import numpy as np
import pyautogui
import time
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def save_face(folder_name, img):
    create_folder(folder_name)
    count = 0
    while True:
        file_name = folder_name + '/' + str(count) + '.jpg'
        if not os.path.exists(file_name):
            cv2.imwrite(file_name, img)
            break
        count += 1


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

last_time = time.time()

folder_name = 'faces'

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                pyautogui.moveTo(x + w / 2, y + h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if time.time() - last_time > 1:
                save_face(folder_name, gray[y:y + h, x:x + w])
                last_time = time.time()

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
