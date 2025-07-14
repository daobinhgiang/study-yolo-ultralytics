from ultralytics import YOLO
import torch
from dataset_loading import CustomizedTrainer
# Load a pretrained model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.train(data="./split_images", trainer=CustomizedTrainer, epochs=1, imgsz=64, device=device, save=True)
print("test")