import cv2
from djitellopy import tello
import cvzone
#  pip freeze > requirements.txt

thres = 0.55
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn.DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            print(classId, conf, box)
            cvzone.cornerRect(img, box)
    except:
        pass
    cv2.imshow("Image",img)
    cv2.waitKey(1)