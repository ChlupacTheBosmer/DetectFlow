from detectflow.config import DETECTFLOW_DIR
import os

VISITORS_MODEL = os.path.join(DETECTFLOW_DIR, 'models', 'visitors.pt')
FLOWERS_MODEL = os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt')
FLOWERS_MODEL_yolov8n = os.path.join(DETECTFLOW_DIR, 'models', 'flowers_yolov8n.pt')
FLOWERS_MODEL_yolov10x = os.path.join(DETECTFLOW_DIR, 'models', 'flowers_yolov10x.pt')

DEFAULT_MODEL_CONFIG = {
    0: {'path': FLOWERS_MODEL_yolov8n,
        'conf': 0.1},
    1: {'path': VISITORS_MODEL,
        'conf': 0.1},
    }
