import cv2
import math
import matplotlib.pyplot as plt
import argparse
import requests
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
faceNet=cv2.dnn.readNet(faceModel,faceProto) 
padding=20
conf_threshold=0.3
net = faceNet
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()
	def get_frame(self):
	    ret, frame = self.video.read()
	    frameOpencvDnn=frame.copy()
	    frameHeight=frameOpencvDnn.shape[0]
	    frameWidth=frameOpencvDnn.shape[1]
	    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

	    net.setInput(blob)
	    detections=net.forward()
	    faceBoxes=[]
	    #cv2.circle(frameOpencvDnn, (480, 321), 60, (255, 0, 0), 4)
	    #cv2.rectangle(frameOpencvDnn,(100,50),(550,300),(0, 255 , 255), 5)
	    #cv2.line(frameOpencvDnn,(250,300), (250,50),(0 ,255, 255), 5)
	    #cv2.line(frameOpencvDnn,(400,300), (400,50),(0 ,255, 255), 5)
	    for i in range(detections.shape[2]):
	        confidence=detections[0,0,i,2]
	        if confidence>conf_threshold:
	            x1=int(detections[0,0,i,3]*frameWidth)
	            y1=int(detections[0,0,i,4]*frameHeight)
	            x2=int(detections[0,0,i,5]*frameWidth)
	            y2=int(detections[0,0,i,6]*frameHeight)
	            faceBoxes.append([x1,y1,x2,y2])
	            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
	    ret, jpeg = cv2.imencode('.jpg', frameOpencvDnn)
	    return jpeg.tobytes()
	    #cv2.imshow("Face Detection", frameOpencvDnn)
