from ultralytics import YOLO
from dataset_loading import CustomizedValidator
model = YOLO("/home/giangdb/Documents/ETC/runs/classify/train5/weights/best.pt")

# Validate the model
metrics = model.val(data='/home/giangdb/Documents/ETC/study-yolo-ultralytics/split_images', validator=CustomizedValidator)  # no arguments needed, dataset and settings remembered
print(metrics.top1)  # top1 accuracy
print(metrics.top5)
