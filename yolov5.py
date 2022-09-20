import sys
import torch

sys.path.insert(0,'./yolov5')
import yolov5

# load pretrained model
model = yolov5.load('yolov5s.pt')
# or load custom model
# model = yolov5.load('train/best.pt')
# set model parameters
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
# perform inference
results = model(img)
# Inference with larger input size
# results = model(img, size=1280)
# inference with test time augmentation
#results = model(img, augment=True)
# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]
scores = predictions[:, 4]
categories = predictions[:, 5]
# x1, y1, x2, y2
# show detection bounding boxes on image
results.show()
# save results into "results/" folder
# results.save(save_dir='results/')