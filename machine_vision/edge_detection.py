import numpy as np
import cv2

def empty(pos):
    pass

image_path = "C:/Users/KAMIC/Desktop/melt pool image_sample/6.jpg"
img =cv2.imread(image_path)

name = "Trackbar"
cv2.namedWindow(name)
cv2.createTrackbar("threshold1", name, 0, 255, empty)
cv2.createTrackbar("threshold2", name, 0, 255, empty)

while True:
    threshold1 = cv2.getTrackbarPos("threshlod1", name)
    threshold2 = cv2.getTrackbarPos("threshlod2", name)

    canny = cv2.Canny(img, threshold1, threshold2)

    cv2.imshow("img", img)
    cv2.imshow(name, canny)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()