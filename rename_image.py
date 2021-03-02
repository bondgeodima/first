import os
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

IMAGE_DIR = 'F:/car_kosmos/new_rgb_tile/tile_128/tif/'
IMAGE_DIR_OUT = 'F:/car_kosmos/new_rgb_tile/tile_128/tif_out/'

for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[2] == 'tif':
        filename_out = filename.split(".")[0] + '_' + filename.split(".")[1] + '.' + filename.split(".")[2]
        old_patch = os.path.join(IMAGE_DIR, filename)
        new_patch = os.path.join(IMAGE_DIR_OUT, filename_out)
        shutil.move(old_patch, new_patch)
        # print("OK")
