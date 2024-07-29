import os

RESOURCES_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(RESOURCES_DIR, "img")
ALT_DIR = os.path.join(RESOURCES_DIR, "alt")

# Create a dictionary ith all icon paths
ICONS = {}
for file in os.listdir(IMG_DIR):
    if file.endswith(".svg"):
        ICONS[file.split(".")[0]] = os.path.join(IMG_DIR, file)

# Create a dictionary with all alternative image paths
ALT_ICONS = {}
for file in os.listdir(ALT_DIR):
    if file.endswith(".svg"):
        ALT_ICONS[file.split(".")[0]] = os.path.join(ALT_DIR, file)

# Create a dictionary with all image paths
IMGS = {}
for file in os.listdir(IMG_DIR):
    if file.endswith(".png"):
        IMGS[file.split(".")[0]] = os.path.join(IMG_DIR, file)

CSS = {}
for file in os.listdir(os.path.join(RESOURCES_DIR, 'css')):
    if file.endswith(".css"):
        CSS[file.split(".")[0]] = os.path.join(RESOURCES_DIR, 'css', file)
