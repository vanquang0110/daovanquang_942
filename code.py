import cv2
import numpy as np
import os

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('yolov3.txt','r') as f:
    classes = f.read().splitlines()

output_folder = "Ket_qua"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


img = cv2.imread('input.jpg')
#img = cv2.resize(img, None, fx=0.5, fy=0.5)
height,width,_=img.shape

blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True, crop=False)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            label = classes[class_id]

            label_folder = os.path.join(output_folder, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            center_x =int(detection[0]*width)
            center_y =int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

            object_img = img[y:y + h, x:x + w]
            object_path = os.path.join(label_folder, f"{label}.jpg")
            cv2.imwrite(object_path, object_img)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes),3))

if len(indexes) == 0:
    print("t")

if len(indexes) > 0:
    print("f")
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(img, (x, y), (x + w, y - 30), color, -1)
        cv2.putText(img, label + " " + confidence, (x, y - 4), font, 2, (255, 255, 255), 2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()