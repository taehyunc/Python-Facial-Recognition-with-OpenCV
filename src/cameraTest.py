#Program to experiment and test opencv
import numpy as np
import cv2

#Use Default Video Camera device
cap = cv2.VideoCapture(0)

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

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()

# Change scale / size of frames function. Default 75
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Continuous usage of the video
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = rescale_frame(frame, percent=150)
    cv2.imshow('frame',frame)
    # Shows camera in gray frame
    # cv2.imshow('gray',gray)

    # Display the resulting frame -- Hit q to exit the application
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything's done, release the capture
cap.release()
cv2.destroyAllWindows()
