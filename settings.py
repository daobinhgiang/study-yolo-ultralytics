from ultralytics import YOLO
from ultralytics import settings

# Update multiple settings
settings.update({"runs_dir": "./results",
                 "weights_dir": "./model_weights",
                 "datasets_dir": "./datasets"})