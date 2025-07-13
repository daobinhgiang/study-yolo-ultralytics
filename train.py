from ultralytics import YOLO
import torch
# Load a pretrained model
model = YOLO("./model_weights/yolo11n-cls.pt")  # load a pretrained model (recommended for training)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = model.train(data="./split_images", epochs=1, imgsz=64, device=device, save=True)