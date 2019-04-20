# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 04:35:37 2018

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 02:25:30 2018

@author: DELL
"""

import cv2
fist_cascade=cv2.CascadeClassifier('haarcascade_fist.xml')
def detect(gray,frame):
    fists=fist_cascade.detectMultiScale(gray,1.7,5)
    for (x,y,w,h) in fists:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Hand',(x+w,y+h), font, 1, (200,255,155), 2, cv2.LINE_AA)
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