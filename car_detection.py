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

COCO_MODEL_PATH_ALL = os.path.join(ROOT_DIR, "car.h5")

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
    IMAGE_DIR = 'F:/sas_out/varshava_256/' + item + '/'

    s = {
        "type": "FeatureCollection",
        "name": "car",
        "crs": {"type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": []
    }

    features = []

    # f = open("F:/car_image_train/out.txt", "a")

    for filename in os.listdir(IMAGE_DIR):
        if filename.split(".")[1] == 'jpg':
            print (filename)
            your_path = os.path.join(IMAGE_DIR, filename)
            file_coord = filename.split(".")[0] + '.jgw'
            file_coord = os.path.join(IMAGE_DIR, file_coord)
            # print(your_path)

            file1 = open(file_coord, 'r')
            Lines = file1.readlines()
            count = 0
            x = 0
            y = 0
            # Strips the newline character
            for line in Lines:
                if count == 0:
                    scale = float(line.strip())
                if count == 4:
                    x_coord = float(line.strip())
                if count == 5:
                    y_coord = float(line.strip())
                count = count + 1

            size = os.path.getsize(your_path)

            image_all = skimage.io.imread(your_path)
            height, width, depth = image_all.shape
            image_all = np.flipud(image_all)

            # Run detection
            results_all = model_all.detect([image_all], verbose=1)

            # Visualize results
            r_all = results_all[0]
            # visualize.display_instances(image_all, r_all['rois'], r_all['masks'], r_all['class_ids'], class_names_all, r_all['scores'])

            i = 0
            for val in r_all['rois']:

                # print(val[0], val[2], val[1], val[3])

                # ---------------------------------------------
                # Идея вырезать картинки не по ббокс а по контуру маски.
                # Будет ли так работать предсказание точнее?

                masks = r_all['masks'][:, :, i]
                ground_truth_binary_mask = masks
                contours = measure.find_contours(ground_truth_binary_mask, 0.5)
                try:
                    hull = ConvexHull(contours[0])
                    polygon = []
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "score": 0.012
                        },
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": []
                        }
                    }
                    for v in hull.vertices:
                        pt = []
                        # print(contours[0][v][0], contours[0][v][1])
                        x = round((x_coord + (int(contours[0][v][1]) * scale)), 4)
                        # y = round(((y_coord + (int(contours[0][v][0]) * scale)) - 1024 * scale), 4)
                        y = round(((y_coord + (int(contours[0][v][0]) * scale)) - height * scale), 4)
                        # x = int(contours[0][v][0])
                        # y = int(contours[0][v][1])
                        pt.append(x)
                        pt.append(y)
                        polygon.append(pt)
                    # print (polygon)
                    pts = np.array(polygon)

                    feature["geometry"]["coordinates"] = [[polygon]]
                    # feature["geometry"]["coordinates"] = [[new_matrix.tolist()]]
                    features.append(feature)

                    i = i + 1
                except Exception as e:
                    print(e)
    # f.close()
    s["features"] = features
    # print(json.dumps(s, indent=4))R
    # with open('F:/car_image_train/data.geojson', 'w') as outfile:
    # f_out = 'F:/Kiev_tile/' + item + '.geojson'
    f_out = 'F:/sas_out/varshava_256/' + item + '.geojson'
    with open(f_out, 'w') as outfile:
        json.dump(s, outfile)
    print("End")
