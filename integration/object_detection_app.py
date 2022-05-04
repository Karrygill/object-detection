from flask import Flask, request, jsonify
import cv2
import numpy
app = Flask(__name__)

import numpy as np
import argparse
import time
import cv2
import os
import sys
from six import StringIO
import requests
import json
from PIL import Image
import jsonpickle as pickle

# from google.colab.patches import cv2_imshow

confthres=0.2
nmsthres=0.4

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    # lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(labels_path).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def get_predection(image, net, LABELS,COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    result = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            print(LABELS[classIDs[i]])
            # result.append([LABELS[classIDs[i]], confidences[i], x, y, x + w, y + h])
            result.append([LABELS[classIDs[i]], x, y, x + w, y + h])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return result, image


def run_detection(image, file_path):
    # load our input image and grab its spatial dimensions
    # image = cv2.imread("test1.jpeg")
    # image = cv2.imread(sys.argv[1])
    labelsPath="darknet/data/coco.names"
    CFG="darknet/cfg/yolov3.cfg"
    Weights="darknet/yolov3.weights"

    Lables=get_labels(labelsPath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)

    result, img = get_predection(image,nets,Lables,Colors)
    # print(result)
    # json_str = json.dumps(result)
    json_str = result
    output = {'data' : json_str}
    url = 'http://192.168.1.106:5002/pose'
    # image = {'image': open('test1.jpeg', 'rb'),
    #          'data': ('data', json.dumps(output), 'application/json')}

    image = {'image': open(file_path, 'rb'),
             'data': ('data', json.dumps(output), 'application/json')}
    # x = requests.post(url, json=output, files=image)
    print(json_str)
    # print(x)
    file_path = file_path
    file_path = os.path.join("", file_path)
    cv2.imwrite(filename="dt"+file_path, img=img)
    # cv2.imshow("Image", img)
    cv2.waitKey()

@app.route('/')
def pose_service():
    return 'Hello, I am object detection service. OK!'

@app.route('/detect', methods = ['POST'])
def do_object_detection():
    file = request.files['image']
    # boxes = request.files['data']
    # d = json.load(boxes)['data']
    # print(d)
    #
    # for x in d:
    #     print(x)

    img = Image.open(file.stream)
    tst = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)

    file_path = os.path.join("", file.filename)
    cv2.imwrite(filename=file_path, img=tst)

    print("------------------------------")
    run_detection(tst, file_path)
    print("------------------------------")
    return 'object detection'

@app.route('/image_acquisition', methods = ['POST'])
def do_image_acquisition():

    data = request.data
    print(data)
    # vision_interface = VisionInterface()
    return "image acquisition done"


@app.route('/detectImage', methods = ['POST'])
def do_object_read():
    timestamp = str(int(time.time()))
    file = request.files['image']
    img = Image.open(file.stream)
    tst = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)

    file_path = os.path.join("", file.filename)
    cv2.imwrite(filename=file_path, img=tst)

    print("receive file name ", file.filename)
    print("output_file_name ", file_path)

    json_str = [[2, 716, 195, 1076, 294]]
    output = {'data' : json_str}

    url = 'http://192.168.1.105:5003/readImage'
    image = {'image': open(file_path, 'rb'),
             'data': ('data', json.dumps(output), 'application/json')}
    requests.post(url, files=image)

    print("------------------------------")
    # run_detection(tst)
    print("------------------------------")
    return 'object detection'


if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5001, debug = True)
