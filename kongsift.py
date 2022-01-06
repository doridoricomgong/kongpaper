import numpy as np
import cv2 as cv

img = cv.imread('./Open/t01/5.jpg')
img = cv.resize(img, dsize=(800, 600), interpolation=cv.INTER_AREA)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('./Open/t01/6.jpg',img)

#filename = './Open/t01/5.jpg'
#img = cv2.imread(filename)
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
#img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('img', img)
#cv2.waitKey()
#cv2.destroyAllWindows()