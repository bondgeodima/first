from PIL import Image
import requests
# from io import BytesIO
import io
import os
from matplotlib import pyplot as plt

import os
import numpy as np
import skimage.io

import sys
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco

from skimage import measure
from scipy.spatial import ConvexHull

import json

IMAGE_DIR_OUT = 'F:/tmp/'
filename_out = 'tmp.jpg'
new_patch = os.path.join(IMAGE_DIR_OUT, filename_out)


class InferenceConfigAll(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # single sign detect
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building


# Root directory of the project
ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

sys.path.append(os.path.join(ROOT_DIR, ""))  # To find local version

# Local path to trained weights file
# single sign detect

COCO_MODEL_PATH_ALL = os.path.join(ROOT_DIR, "car_google.h5")

# IMAGE_DIR = os.path.join(ROOT_DIR, "F:/car_image_train/")

print(COCO_MODEL_PATH_ALL)

configAll = InferenceConfigAll()
# configAll.display()

# Create model object in inference mode.
model_all = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configAll)

# Load weights trained on MS-COCO
model_all.load_weights(COCO_MODEL_PATH_ALL, by_name=True)

class_names_all = ['BG', 'car']

for i in range(242601, 242611):

    # url = 'https://khms0.google.com/kh/v=894?x=242601&y=393871&z=20'
    url = 'https://khms0.google.com/kh/v=894?x='+str(i)+'&y=393871&z=20'

    r = requests.get(url, stream=True)
    im = Image.open(io.BytesIO(r.content))
    nx, ny = im.size

    im2 = im.resize((int(nx * 4), int(ny * 4)), Image.BICUBIC)
    im2.save(new_patch, dpi=(576, 576))

    im3 = Image.open(new_patch)

    # im = Image.open(requests.get(url, stream=True).raw)

    # response = requests.get(url)
    # print (response)
    # im = Image.open(BytesIO(response.content))
    imgplot = plt.imshow(im3)
    # plt.show()

    nx, ny = im3.size
    print(nx, ny)

    image_all = skimage.io.imread(new_patch)
    height, width, depth = image_all.shape

    # Run detection
    results_all = model_all.detect([image_all], verbose=0)

    r_all = results_all[0]
    # visualize.display_instances(image_all, r_all['rois'], r_all['masks'], r_all['class_ids'], class_names_all, r_all['scores'])

