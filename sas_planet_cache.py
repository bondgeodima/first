from maptiler import GlobalMercator
import os

import sqlite3
from sqlite3 import Error

import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
COCO_MODEL_PATH_ALL = os.path.join(ROOT_DIR, "car_google_low.h5")

# IMAGE_DIR = os.path.join(ROOT_DIR, "F:/car_image_train/")

print(COCO_MODEL_PATH_ALL)

configAll = InferenceConfigAll()
# configAll.display()

# Create model object in inference mode.
model_all = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configAll)

# Load weights trained on MS-COCO
model_all.load_weights(COCO_MODEL_PATH_ALL, by_name=True)

class_names_all = ['BG', 'car']

lat1 = 50.351518
long1 = 30.315950

lat2 = 50.530783
long2 = 30.713327

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


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def select_rows(conn, X, Y):
    """
    Query 10 rows
    :param conn:
    :return:
    """
    cur = conn.cursor()
    sql = "SELECT * FROM t WHERE x=" + str(X) + " AND y=" + str(Y)

    # print (sql)

    cur.execute(sql)

    rows = cur.fetchall()

    for row in rows:
        # print(row[7])
        im = Image.open(io.BytesIO(row[7]))
        nx, ny = im.size

        # imgplot = plt.imshow(im)
        # plt.show()

        im2 = im.resize((int(nx * 4), int(ny * 4)), Image.BICUBIC)
        im2.save(your_path, dpi=(576, 576))

        # img = mpimg.imread(your_path)
        # imgplot = plt.imshow(img)
        # plt.show()


def main():

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

            # url = 'https://khms0.google.com/kh/v=894?x='+str(i)+'&y='+str(j)+'&z='+str(z)+''
            database = 'F:/SAS_new/cache_sqlite2/sat/z' + str(z+1) + \
                  '/' + str(i//1024) + '/' + str(j//1024) + '/' + str(i//256) + '.' + str(j//256) + '.sqlitedb'

            # print (database)

            # create a database connection
            conn = create_connection(database)

            with conn:

                select_rows(conn, i, j)
                detect_image(x_coord, y_coord, dx)
                # features.append(feature)

            conn.close()

    s["features"] = features

    f_out = r'F:\car_kosmos\google\detect_car_google_low_model_sasplanet_cache_2.geojson'
    with open(f_out, 'w') as outfile:
        json.dump(s, outfile)

    print("END CALCULATION")


def detect_image(x_coord, y_coord, dx):

    size = os.path.getsize(your_path)

    image_all = skimage.io.imread(your_path)
    height, width, depth = image_all.shape
    image_all = np.flipud(image_all)

    scale = round(dx/height, 10)

    # Run detection
    results_all = model_all.detect([image_all], verbose=0)

    # Visualize results
    r_all = results_all[0]
    # visualize.display_instances(image_all, r_all['rois'], r_all['masks'], r_all['class_ids'], class_names_all,
    #                             r_all['scores'])

    q = 0
    for val in r_all['rois']:

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
                y = round(((y_coord + (int(contours[0][v][0]) * scale)) - height * scale), 4)
                pt.append(x)
                pt.append(y)
                polygon.append(pt)
            # print (polygon)

            feature["geometry"]["coordinates"] = [[polygon]]
            # feature["geometry"]["coordinates"] = [[new_matrix.tolist()]]
            features.append(feature)

            q = q + 1

        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
