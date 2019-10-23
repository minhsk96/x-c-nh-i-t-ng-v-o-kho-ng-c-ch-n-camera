import math
from imutils import paths
import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)
def find_marker(image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 35, 125)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x,y,w,h) in faces:
                image = cv2.rectangle(image,(x,y),(x+w,y+h),(455,0,0),2)
        
        #cv2.imshow('img', edged)
        a=()
        if faces!=a:
            return(faces)
        else:
        	return 0
        	
Line1=400
Line2=350
while(1):
        ret, frame = cap.read()
        marker = find_marker(frame)
        cv2.line(frame,(0,Line1),(640,Line1),(0,255,0),2)
        cv2.imshow("test", frame)
        if not ret:
               break
        k = cv2.waitKey(33)
        if abs((marker[0][2]+marker[0][1])-Line1)<5:
               img1 = frame
               cv2.imshow("img1", img1)
               break
while(1):
        ret, frame = cap.read()
        marker = find_marker(frame)
        cv2.line(frame,(0,Line2),(640,Line2),(0,255,0),2)
        cv2.imshow("test", frame)
        if not ret:
               break
        k = cv2.waitKey(33)
        if abs((marker[0][2]+marker[0][1])-Line1)<5:
               img2 = frame
               cv2.imshow("img2", img2)
               break
while(1):
        k = cv2.waitKey(33)
        if (k == 32):
               break
                
marker1 = find_marker(img1)
marker2 = find_marker(img2)
                                              
print marker1
print marker2
cv2.destroyAllWindows()

focalLength = 700;
DISTANCE = marker1*10/(marker2-marker1)
WIDTH = DISTANCE*marker2/focalLength

print DISTANCE
print WIDTH

def distance_to_camera(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth
        
while(1):
    ret, image = cap.read()
    marker = find_marker(image)
    real_distance = distance_to_camera(WIDTH, focalLength, marker)
    #print(inches)
    cv2.putText(image, "%.2fcm" % real_distance,(image.shape[1] - 300, image.shape[1] - 550), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    
    if cv2.waitKey(1)==ord('q'):
        break;
        
cv2.destroyAllWindows()
cap.release()


