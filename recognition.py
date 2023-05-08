import numpy as np
import cv2
import pickle
import math

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #recognising Images (can use Deep learned models predict, keras, pytorch, tensorflow)
        id_, conf = recognizer.predict(roi_gray)
        
        if conf >= 60 and conf <= 95:
            # print(id_)
            print(labels[id_])
            print(math.floor(conf))
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y-10), font, 0.8, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew ,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
        
        #profile = profile_cascade.detectMultiScale(roi_gray)
        # for (x, y, w ,h) in profile:
            # cv2.rectangle(roi_color,(px,py),(px+pw, py+ph),(255,0,0),2)
            # cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('frame1',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()