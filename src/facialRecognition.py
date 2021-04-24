import numpy as np
import cv2

#Use Default Video Camera device
cap = cv2.VideoCapture(0)
imageType = ".png"

# Different Resolutions to adjust camera capture
# 3 -> width
# 4 -> height
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def changeRes(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()

# Change scale / size of frames function. Default 75
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# Continuous usage of the video
while(True):
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=150)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roiGray = gray[y:y+h, x:x+w]
        # This will save just your face and save into directory
        imgItem = "myFace" + imageType
        # cv2.imwrite(imgItem, roiGray)

        colorRectangle = (255, 0, 0) #BGR 0-255
        strokeRectangle = 2 #Thickness of the rectangle frames
        endCordX = x + w #End coordinates of where face ends in x axis
        endCordY = y + h #End coordinates of where face ends in y axis
        cv2.rectangle(frame, (x,y), (endCordX, endCordY), colorRectangle, strokeRectangle)
    cv2.imshow('frame',frame)

    # Display the resulting frame -- Hit q to exit the application
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything's done, release the capture
cap.release()
cv2.destroyAllWindows()
