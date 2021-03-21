import requests
import os
import io
from PIL import Image

from maptiler import GlobalMercator

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

# COCO_MODEL_PATH_ALL = os.path.join(ROOT_DIR, "car_google.h5")
COCO_MODEL_PATH_ALL = os.path.join(ROOT_DIR, "car_qb.h5")

# IMAGE_DIR = os.path.join(ROOT_DIR, "F:/car_image_train/")

print(COCO_MODEL_PATH_ALL)

configAll = InferenceConfigAll()
# configAll.display()

# Create model object in inference mode.
model_all = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configAll)

# Load weights trained on MS-COCO
model_all.load_weights(COCO_MODEL_PATH_ALL, by_name=True)

class_names_all = ['BG', 'car']

# lat1 = 52.200643
# long1 = 20.974275

# lat2 = 52.203171
# long2 = 20.980568

lat1 = 49.517509
long1 = 25.522612

lat2 = 49.593007
long2 = 25.673538

# Работать надо с 19 зумом гугла
# z = 19

z = 19

g1 = GlobalMercator()

x1, y1 = g1.LatLonToMeters(lat1, long1)
t1 = g1.MetersToTile(x1, y1, z)
t1 = g1.GoogleTile(t1[0], t1[1], z)

x2, y2 = g1.LatLonToMeters(lat1, long2)
t2 = g1.MetersToTile(x2, y2, z)
t2 = g1.GoogleTile(t2[0], t2[1], z)

x3, y3 = g1.LatLonToMeters(lat2, long2)
t3 = g1.MetersToTile(x3, y3, z)
t3 = g1.GoogleTile(t3[0], t3[1], z)

x4, y4 = g1.LatLonToMeters(lat2, long1)
t4 = g1.MetersToTile(x4, y4, z)
t4 = g1.GoogleTile(t4[0], t4[1], z)

tx = [t1[0], t2[0], t3[0], t4[0]]
ty = [t1[1], t2[1], t3[1], t4[1]]

# print (t1)
# print (t2)
# print (t3)
# print (t4)

# print ("")

# dx = [t1[0], t2[0]]
# dy = [t3[1], t2[1]]

# print (dx)
# print (dy)

# url = 'https://khms0.google.com/kh/v=894?x=242601&y=393871&z=20'
IMAGE_DIR_OUT = 'F:/tmp/'
filename_out = 'tmp.jpg'
your_path = os.path.join(IMAGE_DIR_OUT, filename_out)

s = {
    "type": "FeatureCollection",
    "name": "car",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
    "features": []
}

features = []

count = (max(tx) - min(tx))*(max(ty)-min(ty))

print("count tile = " + str(count))

t = 1
for i in range(min(tx), max(tx)):
    for j in range(min(ty), max(ty)):

        print (str(i) + " " + str(j) + " " + str(t) + " from " + str(count))
        t = t+1
        tms_tile = g1.GoogleTile(i, j, z)
        b = g1.TileBounds(tms_tile[0], tms_tile[1], z)

        dx = b[2] - b[0]

        x_coord = b[0]
        y_coord = b[3]

        url = 'https://khms0.google.com/kh/v=894?x='+str(i)+'&y='+str(j)+'&z='+str(z)+''

        r = requests.get(url, stream=True)

        im = Image.open(io.BytesIO(r.content))
        nx, ny = im.size

        im2 = im.resize((int(nx * 4), int(ny * 4)), Image.BICUBIC)

        scale = round(dx / (nx * 4), 10)

        if im2:

            im2.save(your_path, dpi=(288, 288))
            # im2.save(your_path, dpi=(576, 576))
            # im2.save(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.jpg', dpi=(576, 576))

            # im3 = Image.open(new_patch)

            """
            f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.jgw', "w+")
            f.write(str(scale) + '\r\n')
            f.write(str("0.0000000000") + '\r\n')
            f.write(str("0.0000000000") + '\r\n')
            f.write("-" + str(scale) + '\r\n')
            f.write(str(b[0]) + '\r\n')
            f.write(str(b[3]) + '\r\n')
            f.close()
    
            f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.prj', "w+")
            f.write('PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                    'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
                    'UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],'
                    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
                    'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],'
                    'PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]')
            f.close()    
            """

            size = os.path.getsize(your_path)

            image_all = skimage.io.imread(your_path)
            height, width, depth = image_all.shape
            image_all = np.flipud(image_all)

            # Run detection
            results_all = model_all.detect([image_all], verbose=0)

            # Visualize results
            r_all = results_all[0]

            q = 0
            for val in r_all['rois']:

                # print (r_all['scores'])

                # print(val[0], val[2], val[1], val[3])

                # ---------------------------------------------
                # Идея вырезать картинки не по ббокс а по контуру маски.
                # Будет ли так работать предсказание точнее?

                masks = r_all['masks'][:, :, q]
                ground_truth_binary_mask = masks
                contours = measure.find_contours(ground_truth_binary_mask, 0.5)
                try:
                    hull = ConvexHull(contours[0])
                    polygon = []
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "scores": 0.012
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

                    q = q + 1
                except Exception as e:
                    print(e)

s["features"] = features

f_out = 'F:/tmp/detect_car_google_model.geojson'
with open(f_out, 'w') as outfile:
    json.dump(s, outfile)

print("END CALCULATION")


