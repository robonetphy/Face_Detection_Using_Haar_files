# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 02:25:30 2018

@author: DELL
"""

import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile_0.xml')

def detect(gray,frame):
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi1=gray[y:y+h,x:x+w]
        roi2=frame[y:y+h,x:x+w]
        eye=eye_cascade.detectMultiScale(roi1,1.1,22)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi2,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Person',(x+w,y+h), font, 1, (200,255,155), 2, cv2.LINE_AA)
        smile=smile_cascade.detectMultiScale(roi1,1.7,22)
        for (sx,sy,sw,sh) in smile:
             cv2.rectangle(roi2,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame
        
video=cv2.VideoCapture(0)

while True:
    _,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
    canvas= detect(gray,frame)
    cv2.imshow('Face Recognition',canvas)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()