import os
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

IMAGE_DIR = r'F:\car_kosmos\google\tile_256'
IMAGE_DIR_OUT = r'F:\car_kosmos\google\tif'

for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[2] == 'tif':
        print (filename)
        filename_out = filename.split(".")[0] + '_' + filename.split(".")[1] + '.' + filename.split(".")[2]
        old_patch = os.path.join(IMAGE_DIR, filename)
        new_patch = os.path.join(IMAGE_DIR_OUT, filename_out)
        shutil.move(old_patch, new_patch)
print("END")
