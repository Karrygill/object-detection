# object-detection
Contains code and data files related to the object detection tasks

# Object Detection Documentation

# Tasks involved : 
1. Data Collection
2. Image labeling and augmentation
3. Model Training
4. Model Detection
5. Integration with Pose Estimation
6. Deploying the model


# Data Collection
- The data used for training object detection consisted of synthetic dataset from Nvidia Research and custom images collected in the lab.
- The Nvidia research dataset and lab dataset is uploaded here : https://drive.google.com/drive/folders/1Q3rXiOCSkA2LaQ1IMAddENQM27RHNxSi?usp=sharing
- The dataset consists of images and annotations for those images.
- The annotation is a text file which is the required format for the YOLO model.
- You can convert it to XML or JSON file for other object detection models if required in future.

# Image labeling and augmentation
- Used the Labelmg tool to annotate the collected images.
- Follow this link on how to use the tool : https://github.com/tzutalin/labelImg 
- We augmented the dataset too by rotating, flipping, changing brightness, contrast, cropping and zooming. 
- Used this project for augmentation : https://github.com/mdbloice/Augmentor

# Model training
I used the YOLOv5 model for object detection tasks. You can clone the latest code from this repo - https://github.com/ultralytics/yolov5 

STEPS for running the model on custom dataset on CV lab servers : 

- You can run the model on google colab or local machine to test out things. If the dataset is huge then raise a request to get access to CV lab GPU servers : https://docs.google.com/forms/d/1PQRbbvMTjkoLRdwDWx5HQmRc4ZSE85cmpZ0bnVAWLws/edit
- Once you get the access, follow this document to understand how to use the CV lab servers : https://docs.google.com/document/d/1tj7NOfMbjACsgmSZQXNQvfQvrzLEyH06nftPb-O3aRc/edit
- I created a conda environment in the BigBang server to run the yolo model. Use the latest python version preferably.
- Follow this link on how to run the model : https://blog.paperspace.com/train-yolov5-custom-data/
- Tune the parameters according to the dataset and make the changes in the config file as mentioned in the above link. I trained our dataset on yolo medium network for 20 epochs with MAP@.95 = 0.784 and MAP@0.5 = 0.958
- Once the training is complete, you can check the results in the run folder.
- Download the best weight generated during the training. It will be used for the detection task later.

# Model detection
- Once the training is complete, you can use the best weights generated during training for the detection task on any new image.
- Find the corresponding code here : 

# Integration with Pose Estimation
- Built a simple Flask App in Python to run the object detection feature. 
- Once the object detection API is called, it runs the detection task and sends the bounding box and class label to the pose estimation feature.
- Pose estimation exposes an API to get bounding box, class label and image from object detection
- We integrated the YOLOv3 version with Pose Estimation. Find the corresponding code here :
- Follow the same steps to integrate YOLOv5 with Pose Estimation.
- You will have to deploye object detection and pose estimation apps seperately and use the corresponding IP addresses to call the APIS.

# Deploying the model
- You can deploy the Flask App on servers. 
- Parallely, we explored deploying the individual flask apps using Docker images. Find the corresponding documentation here : https://medium.com/@shreyash-hisariya/deploying-deep-learning-models-using-docker-yolo-afd596e56d7a
