import numpy as np
import cv2
cap = cv2.VideoCapture(0)  
while(1):
    _, image = cap.read()
    image = cv2.GaussianBlur(image,(3,3),0)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.9, 100)
    if circles is not None:
            circles = np.round(circles[0, :]).astype("int")    
            for (x, y, r) in circles:                   
                    cv2.circle(image, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    cv2.imshow("output", image)
    if cv2.waitKey(1)==ord('q'):
        break           
cv2.destroyAllWindows()
cap.release()
