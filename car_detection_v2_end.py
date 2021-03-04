"""
Реализация идеи подачи на обработку нескольких изображений за раз
"""

import os
import glob
import numpy as np
import skimage.io

import sys
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco

from skimage import measure
from scipy.spatial import ConvexHull

import json


class InferenceConfigAll(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    # BATCH_SIZE = 12
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

"""
array = ['80-0825', '80-0826', '80-0827', '80-0828', '80-0829', '80-0830', '80-0831',
         '80-0873', '80-0874', '80-0875', '80-0876', '80-0877', '80-0878', '80-0879',
         '80-0921', '80-0922', '80-0923', '80-0924', '80-0925', '80-0926',
         '80-0969', '80-0970', '80-0971', '80-0972']
"""

array = ['jpg']

for item in array:
    # IMAGE_DIR = os.path.join(ROOT_DIR, "F:/Kiev_tile/80-0734/")
    # IMAGE_DIR = 'F:/Kiev_tile/' + item + '/'
    IMAGE_DIR = 'F:/Poland/image_dop_2/' + item + '/'

    list_file = glob.glob(IMAGE_DIR + '*.jpg')
    print("File in list = " + str(len(list_file)))

    s = {
        "type": "FeatureCollection",
        "name": "car",
        "crs": {"type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": []
    }

    features = []

    # f = open("F:/car_image_train/out.txt", "a")

    for f in range(0, len(list_file), 3):
        list_slice = list_file[f:f + 3]
        image_list = []
        print(list_slice)
        for i in list_slice:
            im = skimage.io.imread(i)
            height, width, depth = im.shape
            im = np.flipud(im)
            image_list.append(im)
        # print(image_list)
        assert len(
            image_list) == configAll.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        results_all = model_all.detect(image_list, verbose=0)
        # r_all = results_all[0]
        # print(results_all)


