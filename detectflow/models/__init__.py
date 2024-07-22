from detectflow.config import DETECTFLOW_DIR
import os

VISITORS_MODEL = os.path.join(DETECTFLOW_DIR, 'models', 'visitors.pt')
FLOWERS_MODEL = os.path.join(DETECTFLOW_DIR, 'models', 'flowers.pt')
FLOWERS_MODEL_yolov10x = os.path.join(DETECTFLOW_DIR, 'models', 'flowers_yolov10x.pt')
FLOWERS_MODEL_yolov9e = os.path.join(DETECTFLOW_DIR, 'models', 'flowers_yolov9e.pt')

DEFAULT_MODEL_CONFIG = {
    0: {'path': FLOWERS_MODEL,
        'conf': 0.1},
    1: {'path': VISITORS_MODEL,
        'conf': 0.1},
    }
