##FOR CLASSIFICATION##
from ultralytics import YOLO


from roboflow import Roboflow
rf = Roboflow(api_key="pn95lzVYdDCGuSMP0gRO")
project = rf.workspace("drone-bg6aq").project("drone-c8l5j")
version = project.version(12)
dataset = version.download("yolov8") 



model = YOLO("yolov8n.pt")
model.train(data = "//home/kgx/drone/src/drone-12/data.yaml",save = True,epochs = 175,batch = 16)


model1 = YOLO("/home/kgx/drone/runs/detect/train15/weights/80_175epochs.pt")

model1.track(source = "/home/kgx/drone/data/dataset2.mp4" ,save=True)
