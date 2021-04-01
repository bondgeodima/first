import os
import skimage.io
import json

ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, r'F:\car_kosmos\google_v2\jpg')

s = {
    "type": "FeatureCollection",
    "name": "car",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3785" } },
    "features": []
}

features = []

for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[1] == 'jpg':
        # print (filename)
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

        # size = os.path.getsize(your_path)
        img = skimage.io.imread(your_path)
        height, width, depth = img.shape
        x_min = x_coord
        y_min = y_coord - scale * height
        x_max = x_coord + scale * height
        y_max = y_coord
        polygon = []
        feature = {
            "type": "Feature",
            "properties": {
                "score": filename.split(".")[0]
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": []
            }
        }
        polygon = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]
        feature["geometry"]["coordinates"] = [[polygon]]
        features.append(feature)
        print (filename, x_min, y_min, x_max, y_max)

s["features"] = features
# print(json.dumps(s, indent=4))
# with open('F:/car_image_train/data.geojson', 'w') as outfile:
with open(r'F:\car_kosmos\google_v2\data_grid_2019_256.geojson', 'w') as outfile:
    json.dump(s, outfile)
print("End")

