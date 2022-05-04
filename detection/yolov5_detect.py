import numpy as np
import argparse
import time
import cv2
import os
import sys
import torch
from PIL import Image
# from google.colab.patches import cv2_imshow

confthres=0.0
nmsthres=0.0


def main():
    # load our input image and grab its spatial dimensionsn
    # image = cv2.imread("/Users/shreyashhisariya/Downloads/ycb_dataset/augmented/data/test/images/9_mixed.png")
    image = Image.open('/Users/shreyashhisariya/Downloads/ycb_dataset/augmented/data/test/images/4_cheezit_box.png')
    # image = cv2.imread(sys.argv[1])

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/shreyashhisariya/Downloads/ycb_dataset/augmented/train/exp10/weights/best.pt')
    output = model(image)
    output.print()
    output.show()
    print('\n', output.pandas().xyxy[0])
    # output.pandas().xyxy[0]
    # cv2.imshow("Image", res)
    # cv2.waitKey()

if __name__== "__main__":
  main()
