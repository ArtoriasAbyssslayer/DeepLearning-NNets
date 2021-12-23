# draw and write on images
import cv2 as cv
import numpy as np

#drawing on images
#create blank images
blank = np.zeros((500, 500), dtype='uint8')
img = cv.imread("C:/Users/harry/Pictures/wallpapers/pf.jpg")

cv.imshow('pf', img)

# 1. Paint the image a certain colour
blank[200:300, 300:400] = 255, 0, 0
cv.imshow()
cv.waitKey(0)
