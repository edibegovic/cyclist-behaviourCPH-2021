import cv2
import torch
import pandas as pd
import sys
import easygui
import pickle

file_name = input("Please enter file name to use.")
camera = input("Please enter camera used.")

model = torch.hub.load("ultralytics/yolov5", "yolov5x6", force_reload=False)

cap = cv2.VideoCapture(easygui.fileopenbox(msg="Choose video file."))
pickle.load(file)
camera_parameters = pickle.load(file)
K = camera_parameters["K"]
d = camera_parameters["d"]

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 1)

mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

model.classes = [1] #1 for bikes

df = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "confidence", "frame_id"])

count=0
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while(cap.isOpened()):
    sys.stdout.write("\r" + f"{count} frame of {length} - {round((count/length)*100, 2)} %")
    sys.stdout.flush()
    ret, frame = cap.read()
    if ret == False:
        break
    newimg = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    results = model(newimg, size=1280) 
    count+=1
    data = {"xmin": results.pandas().xyxy[0]["xmin"], "ymin": results.pandas().xyxy[0]["ymin"],
            "xmax": results.pandas().xyxy[0]["xmax"], "ymax": results.pandas().xyxy[0]["ymax"],
            "confidence": results.pandas().xyxy[0]["confidence"], "frame_id": count}

    temp = pd.DataFrame(data=data)
    df = pd.concat([df, temp])
 
cap.release()
cv2.destroyAllWindows()

df.to_pickle(f"{file_name}.pickle")

img = cv2.imread("/content/inter_s7.jpg")
newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imwrite("/content/drive/MyDrive/straight_inter_s7.jpg", newimg)