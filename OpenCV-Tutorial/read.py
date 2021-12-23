import cv2 as cv

#---------------------read image basic example----------------

img = cv.imread("C:/Users/harry/Pictures/wallpapers/pf.jpg")

cv.imshow('DarkSideOfTheMoon-PF', img)

cv.waitKey(0)

#---------------------read video example--------------

capture = cv.VideoCapture('G:\Cake Streams/cake.mkv')

while True:
    # capture.read() returns an tuple [isTrue, frame]
    isTrue, frame = capture.read()
    cv.imshow('Cake Stream', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cv.release()
cv.destroyAllWindows()
