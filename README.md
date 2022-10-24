import numpy as np
import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    image = cap.read()
    image = cv2.flip(image, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
   
    if len(result) > 0:
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        cv2.rectangle(image,
                       (bounding_box[0], bounding_box[1]),
                       (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                       (0,155, 255),
                       2)
        cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['right_eye']), 2, (0,155, 255), 2)
        cv2.circle(image,(keypoints['nose']), 2, (0,155, 255), 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, (0,155, 255), 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, (0,155, 255), 2)

        cropped = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                        bounding_box[0]:bounding_box[0]+bounding_box[2]]
   
        cv2.imshow('video',cropped)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cv2.imwrite('screenshots/c2-1.jpg',image)
cap.release()
cv2.destroyAllWindows()
