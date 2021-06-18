import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
tipIds = [4,8,12,16,20]

while True:
  success, img = cap.read()
  img = detector.findHands(img)
  lmList = detector.findPosition(img,draw = False)

  if(len(lmList)!=0):
    fingers = []
    for id in range(0,5):
      if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
        fingers.append(1)
      else:
        fingers.append(0)        

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if fingers.count(1) == 2 and fingers[1] == 1:

      mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
      mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))

      ## Merge the mask and crop the red regions
      mask = cv2.bitwise_or(mask1, mask2 )
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(img,img, mask= mask)
      img = res

    if fingers.count(1)==3 and fingers[2] == 1:

      # define range of green color in HSV
      lower_green = np.array([34, 177, 76])
      upper_green = np.array([255, 255, 255])

      # Threshold the HSV image to get only blue colors
      mask = cv2.inRange(hsv, lower_green, upper_green)
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(img,img, mask= mask)
      img = res

    if fingers.count(1)==4 and fingers[3] == 1:
      #Blue
      lower_blue = np.array([110,50,50])
      upper_blue = np.array([130,255,255])
      mask = cv2.inRange(hsv,lower_blue, upper_blue )
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(img,img, mask= mask)
      img = res
    

  cv2.imshow("Image",img)
  key = cv2.waitKey(1)    

  if key == ord('q'):
    break

