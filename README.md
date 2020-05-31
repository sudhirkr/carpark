# carpark
Detection of empty car parking using YOLOv3 model. It takes feed from carparking cctv camera and does detection on the frames. It is integrated with a public slack channel. Results will be sent to the slack channel as well as it will be stored in local directory.

# config file
Modify the parking coordinates in section "parking_slots" of config.yml. Right now we have 6 parking slots and it lists the fixed co-ordinates of those parking slots.

# yolov3.weights
Model will require yolov3.weights. Its size is 237M so uploaded on Google drive. Download by the link given below and keep it in carpark/yolo-coco directory
https://drive.google.com/file/d/1De9ZSYpkMJrFC-78xhL0QuGfFeFw7QHe/view?usp=sharing

# run
python carpark.py --video videos/test_video.mp4 --yolo yolo-coco/

# results
Output will be stored in carpark/result directory.

# slack channel
Output is sent to a public slack channel. To know more about slack channel app, here is the link by which secured slack app was made.
https://github.com/slackapi/python-slackclient/tree/master/tutorial

# Links referred
Calculate Intersection over Union: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
Object detection using YOLO and OpenCV: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
SSD code: https://honingds.com/blog/ssd-single-shot-object-detection-mobilenet-opencv/

Data augmentation and Image Classifier:

https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
