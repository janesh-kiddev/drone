##FOR SEGMENTATION##
from ultralytics import YOLO


from roboflow import Roboflow
rf = Roboflow(api_key="hnHNys0Tcv7vTpQI8Xaf")
project = rf.workspace("seg-uorfz").project("segmentation_drone")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolov8n-seg.pt")
model.train(data = "/home/kgx/drone/src/segmentation_drone-1/data.yaml",save = True,epochs = 100)

#model1 = YOLO("/home/kgx/drone/runs/segment/train/weights/best.pt")
#results = model1.track(source = "/home/kgx/drone/classify/data/dataset2.mp4",save = True) 
