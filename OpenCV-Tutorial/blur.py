import cv2 as cv
img = cv.imread("C:/Users/harry/Pictures/wallpapers/pf.jpg")

# application of gaussian blurring

blur = cv.GaussianBlur(img, (7, 7))
