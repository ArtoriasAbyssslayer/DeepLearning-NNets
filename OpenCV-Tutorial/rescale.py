# Rescale video and image example to reduce computational load
# best practice downscale the video == reduce the width and the height
# downsampling is what is working

import cv2 as cv
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)  # cv.resize(image,[new width, new height], downsampling algorithm)


capture = cv.VideoCapture('G:\Cake Streams/cake.mkv')

while True:
    isTrue, frame = capture.read() # capture.read() returns an tuple [isTrue, frame]
    frame_resized = rescaleFrame(frame,scale= 0.2)
    cv.imshow('Cake Stream', frame)
    cv.imshow('cake stream resized', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cv.release()
cv.destroyAllWindows()
