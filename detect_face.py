import math
from imutils import paths
import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)
def find_marker(image):
        face_cascade = cv2.CascadeClassifier('D:\F\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x,y,w,h) in faces:
                image = cv2.rectangle(image,(x,y),(x+w,y+h),(455,0,0),2) 
        #cv2.imshow('img', image)
        b= len(faces)
        a=()
        c=()
        if faces==a:
                return(a)
        else:
                for i in range(b):
                        c += ((faces[i][0],faces[i][1],faces[i][2],faces[i][3]),)

        return(c)
                
def distance_to_camera(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth

def check (d,l,f):
        return(d*l)/f


KNOWN_DISTANCE = 50 # khoang cach tu doi tuong den camera ( for cm )

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 16 # khoang cach chieu rong cua mat nguoi
while(1):
        ret, frame = cap.read()
        cv2.imshow("test", frame)
        if not ret:
                break
        k = cv2.waitKey(1)

        if k%256 == 32:
        # SPACE pressed
                img = frame
                break

#cv2.imshow("img", img)
marker = find_marker(img)
#print( marker[0][2] )
focalLength = (marker[0][2] * KNOWN_DISTANCE) / KNOWN_WIDTH
#print(focalLength)
while(1):
        empty=()
        ret, image = cap.read()
        x,y,__ = image.shape
        cv2.line(image,((int(y/2)),0),((int(y/2)),x),(255,0,0))
        marker = find_marker(image)
        if marker != empty:
                b,a,c,d = marker[0][0],marker[0][1],marker[0][2],marker[0][3]
                #print(a,b,c,d)
                y1,x1 = (int((2*a+c)/2)), (int((2*b+d)/2))
                cv2.circle(image,(x1,y1),10,(0,0,255),-1)
                '''delta = - int(y/2) + x1
                if delta > 0:
                       print('turn left')
                elif delta < 0:
                       print('turn right')
                else:
                       print("don't move")'''
        d=len(marker)
        for i in range(d):
                inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[i][2])
                cv2.putText(image, "%.2fcm" % (inches ),(marker[i][0], marker[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                #print(inches)
                cv2.imshow("image", image)
        if cv2.waitKey(1)==ord('q'):
                break;
 
cv2.destroyAllWindows()
cap.release()
    




