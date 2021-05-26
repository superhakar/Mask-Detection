import cv2
import numpy as np 
import argparse
import time
import os

import torch
import torchvision
from torchvision import transforms

from PIL import Image

base_dir = os.path.dirname(__file__)
maskDetect_path = os.path.join(base_dir + r'model_data/final_model.pth')

#loading face detection model and mask detection model
print("loading models")
maskDetectModel = torch.load(maskDetect_path,map_location=torch.device('cpu'))
maskDetectModel.eval()

device = torch.device("cpu")
maskDetectModel.to(device)

#Load yolo
def detect_face(img):
    faces=[]
    positions=[]
    face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    f=face_clsfr.detectMultiScale(img,1.1,2)
    for (x,y,w,h) in f:
        face_img=img[y:y+h,x:x+w]
        faces.append(face_img)
        positions.append((x,y,x+w,y+h))
    return faces,positions

def detect_mask(faces):
    predictions = []
    image_transforms = transforms.Compose([transforms.Resize(size=(244,244)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if (len(faces)>0):
        for img in faces:
            img = Image.fromarray(img)
            img = image_transforms(img)
            img = img.unsqueeze(0)
            prediction = maskDetectModel(img)
            prediction = prediction.argmax()
            predictions.append(prediction.data)
    return predictions

def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x+w-75, y+20), font, 0.5, color, 1)

    
	(faces,postions) = detect_face(img)
	predictions=detect_mask(faces)
	for(box,prediction) in zip(postions,predictions):
		(startX, startY, endX, endY) = box
		label = "Mask" if prediction == 0 else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0,0,255)
		cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(img,(startX, startY),(endX, endY),color,2)

	cv2.imshow("Image", img)
		

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))

	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		writer.write(frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
	writer.release()



if __name__ == '__main__':
	
	#image_detect('./images/00018_Mask.jpg')
	#start_video('./videos/v1.mp4')

	webcam_detect()
	
	cv2.destroyAllWindows()
