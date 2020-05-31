import cv2
import argparse
import time
import os
import imutils
import numpy as np
import yaml
import copy
import json
from slacker import Slacker
import logging

# config file path
CONFIG_FILE="config.yml"

# JSON extract of configs
config = None


"""[summary]
    load coordinates from yaml file. Read the coordinated of the parking slots
Returns:
    [list] -- [parking coordinates]
"""
def get_parking_slots():
    car_parking = []
    
    # get the pre marked parking slots from config file
    slots = config["parking_slots"]
    for s in slots:
        car_parking.append((s["id"], s["coordinates"]))
    print("Total parking slots: ", len(car_parking))
    return car_parking

def send_message_to_slack(response_json, image):
    start_msg = time.time()
    user_token = config["slack_token"]
    channel_name = config["slack_channel"]
    slack = Slacker([user_token])
    response_json = str(response_json)
    slack.files.upload(channels = channel_name, file_= image, initial_comment = response_json)

"""[summary]
"""
def iou(box1, car_parking):
    intersections = []
    parkings = []
    start_iou = time.time()
    for cars in car_parking: 
        box2 = cars[1]
        slots = cars[0]
        parkings.append(box2)
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        #compute area of intersection rectangle
        inter_area = max(0, y2-y1+1)* max(0, x2 - x1+1)
        if inter_area == 0:
            intersections.append(0)
            continue

        #compute area of prediction and ground truth rectangles
        box1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
        box2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = box1_area + box2_area - inter_area
        intersection = inter_area / float(union_area)
        intersections.append(intersection)
        
        
    idx_max = intersections.index(max(intersections))
    total_iou = time.time() - start_iou
#    print("Time to calculate IoU = {:.5f} ".format(total_iou))
#    print("--------------------------------------------------------------------")
    return car_parking[idx_max], intersections[idx_max]

                
"""[summary]
"""
def evaluate_parking(frame, car_boxes, car_parking, frameid):
    result_dir = config["result_dir"]
    image_name = result_dir + "frame%d.jpg"%frameid
    occupied = []
    occ = 0
    emp = 0
    start_1 = time.time()
    for car in car_boxes:
        intersection = config["intersection_value"]
        parking, intersection = iou(car, car_parking)
        if intersection > 0.2:
            occupied.append(parking)
            car_parking.remove(parking)             
    
    for o in occupied:
        cv2.rectangle(frame, (o[1][0], o[1][1]), (o[1][2],o[1][3]), (0,0,255),5)              
        cv2.putText(frame, "Parking slot: "+str(o[0]), (o[1][0],o[1][1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)
        cv2.imwrite(image_name, frame) 
        occ += 1
    
    occupied.sort()
    print("Occupied slots: ",[x[0] for x in occupied])
    X = [x[0] for x in occupied]
    cv2.putText(frame,"Occupied slots: %d"%occ, (20,90), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 5)  
    cv2.imwrite(image_name,frame) 
    
    for e in car_parking:
        cv2.rectangle(frame, (e[1][0],e[1][1]), (e[1][2],e[1][3]), (0,255,0),5)              
        cv2.putText(frame, "Parking slot: "+str(e[0]), (e[1][0], e[1][1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)
        cv2.imwrite(image_name, frame) 
        emp += 1
    cv2.putText(frame,"Empty slots: %d"%emp, (20,140), cv2.FONT_HERSHEY_SIMPLEX, 
                2.0, (0,255,0), 5)  
    cv2.imwrite(image_name, frame) 
    car_parking.sort()
    print("Empty slots: ", [y[0] for y in car_parking])
    print("-"*80)
    Y = [y[0] for y in car_parking]
    total_1 = time.time() - start_1
    logging.debug("Total time to identify and print info on image: {:.5f} ".
                  format(total_1))
    return X, Y, image_name


def print_output_json(slot_no, occ):
    status = 0
    if slot_no in occ:
        status = 1
    display = {'ParkingSlot': slot_no, 'status': status}
    display1 =  json.dumps(display)
    return display
    
   

"""[summary]
    Configurations are in yaml file. Load the config file.
    Returns:
        [JSON] -- []
"""
def open_yaml():
    global CONFIG_FILE 
    with open(CONFIG_FILE, "r") as data:
        try:
            config = yaml.load(data, Loader=yaml.FullLoader)
            return(config)
        except yaml.YAMLError as exc:
            print(exc)
            os.exit(1)
        except Exception as e:
            print(e)
            os.exit(1)


"""[summary]
"""
def detect_cars(layerOutputs, W, H, frameid, frame): 
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        '''
	    loop over each of the detections, extract the class ID and 
        confidence (i.e., probability) of the current object detection
        '''
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            '''
            filter out weak predictions by ensuring the detected probability is 
            greater than the minimum probability
            '''
            if confidence > args["confidence"]:
                
                '''
			    scale the bounding box coordinates back relative to the size of 
                the image, keeping in mind that YOLO actually returns the 
                center (x, y)-coordinates of the bounding box followed by 
                the boxes' width and height
                '''
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                '''
			    use the center (x, y)-coordinates to derive the top and left 
                corner of the bounding box
                '''
                X1 = int(centerX - (width / 2))
                Y1 = int(centerY - (height / 2))
                X2 = int(centerX + (width / 2))
                Y2 = int(centerY + (height /2))
			    
                '''
                update our list of bounding box coordinates, 
                confidences,and class IDs
                '''
                boxes.append([X1, Y1, X2, Y2])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    # ensure at least one detection exists
    if len(idxs) > 0:
	    # loop over the indexes we are keeping
        nmsboxes = []
        for i in idxs.flatten():
            nmsboxes.append(boxes[i])
            (x, y) = (boxes[i][0],boxes[i][1])
            (X, Y) = (boxes[i][2],boxes[i][3])
        
        '''    
        # extract the bounding box coordinates
        draw a bounding box rectangle and label on the image
        cv2.imwrite(image_name, frame)
        '''
    return nmsboxes

#detecting objects
def yolo_detector(args):
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("-"*30, "CAR PARKING SYSTEM", "-"*30)
    logging.info("loading YOLO from disk")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    '''
    load our input image and grab its spatial dimensions
    determine only the output layer names that we need from YOLO
    '''
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
   
    # input video declared
    logging.info("Capturing video...")
    
    reader = cv2.VideoCapture(args["video"])
    seconds = config["status_interval"]
    fps = reader.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".
           format(fps))
    
    # read the first frame
    ret, frame = reader.read()
    #get multiplier that is fps * number of secs
    (H, W) = frame.shape[:2]
    dim = (W, H)
    frameid = 0
    
    # determine total number of frames in a video        
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
        total = int(reader.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        total = -1
   
    # get the coordinates of parking slots 
    car_parking = get_parking_slots()

    count = 0    
    while True:
        count += 1
        if count % 10 != 0:
            continue
        for skip in range(int(fps*seconds)):
            reader.read()
        ret, frame = reader.read()
        logging.info("Video status: {}".format(ret))
        
        if ret == False:
            break
        
        '''
        construct a blob from the input image and then perform a forward pass 
        of the YOLO object detector giving us our bounding boxes and 
        associated probabilities
        '''
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), 
                                     swapRB=True, crop=False)
        net.setInput(blob)
                
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        
        logging.info("YOLO took {:.6f} seconds".format(end - start))
        print("-"*80)
        frameid += 1
        print("Frame id: ", frameid)
                    
        car_parking_copy = copy.deepcopy(car_parking)
        
        '''
        detect all the objects (car) in the frame using YOLO detector. Get the 
        coordinates of the detected car. 
        '''
        car_boxes = detect_cars(layerOutputs, W, H, frameid, frame)
        
        '''
        compare the corordinates from the YOLO detector with the coordinates 
        from the config file. coordinates in the config file are static 
        coordinates.
        '''
        occupied_slots, empty_slots, image  = \
                    evaluate_parking(frame, car_boxes, car_parking_copy,frameid)
        
        #response_json is a list which will contain the dictionary of parking slots
        response_json = []
        for j in car_parking:
            response = print_output_json(j[0], occupied_slots)
            logging.debug(response)
            response_json.append(response)
        send_message_to_slack(response_json, image)
        #return response


if __name__ == "__main__":
    # arguments passed in command line
    logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video",required=True, help="path to video")
    ap.add_argument("-y", "--yolo", required=True, 
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, 
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.8, 
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # fetch the configs from config file and populate inmory structure
    config = open_yaml()
    yolo_detector(args)
    reader.release()
    cv2.destroyAllWindows()
