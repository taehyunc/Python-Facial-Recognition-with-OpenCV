import os
import numpy as np
import cv2
import pickle
from PIL import Image


# Directory of the current local src directory
baseDir = os.path.dirname(os.path.abspath(__file__))
# Directory of the current local directory where face images are stored
imageDir = os.path.join(baseDir, "userFaceImages")
imageTypes = ["png", "jpg", "jfif"]
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
currentId = 0
labelIds = {}
yLabels = []
xTrain = []

#Iterate through the files in image directory
for root, dirs, files in os.walk(imageDir):
    for file in files:
        if filter(file.endswith, imageTypes):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            if not label in labelIds:
                labelIds[label] = currentId
                currentId +=1
            id = labelIds[label]
            # Convert images into grayscale and numbers into numpy array
            pilImage = Image.open(path).convert("L") #GrayScale
            size = (550, 550)
            finalImage = pilImage.resize(size, Image.ANTIALIAS)
            imageArray = np.array(finalImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id)
# Use Pickle to save label ids
with open("labels.pickle", 'wb') as file:
    pickle.dump(labelIds, file)

# Train OpenCV Recognizer. Train the data given xTrain and yLabels (Convert to numpy array), then save into "trainner.yml" file
recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainner.yml")
