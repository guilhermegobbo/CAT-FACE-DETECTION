import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_rgb = cv2.imread('gatothumb.jpg') # original image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # image on gray scale

cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml') # cat cascade

faces = cat_cascade.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=5)

for (x,y,w,h) in faces: # x-coordinate, y-coordinate, width and height of the rectangle
    cv2.rectangle(img_rgb, (x,y), (x+w,y+h), (255,255,0),2)

cv2.imwrite('output_image.jpg', img_rgb)
cv2.imshow('Cat Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()