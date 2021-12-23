import cv2 as cv

img = cv.imread("C:/Users/harry/Pictures/wallpapers/pf.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(10000)  # delay function
