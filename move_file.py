import os
import numpy as np
import skimage.io

import shutil

ROOT_DIR = os.path.abspath("../")

i = 1

"""
IMAGE_DIR = os.path.join(ROOT_DIR, "E:/Kiev/")
# SORT BY FOLDER
for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[1] == 'tif':
        old_patch = os.path.join(IMAGE_DIR, filename)
        NEW_IMAGE_DIR = os.path.join(ROOT_DIR, "E:/Kiev_folder/")
        NEW_IMAGE_DIR = NEW_IMAGE_DIR + filename[0:7] + '/'
        new_patch = os.path.join(NEW_IMAGE_DIR, filename)
        shutil.move(old_patch, new_patch)
        print (i)
        i = i + 1
"""

"""
IMAGE_DIR = os.path.join(ROOT_DIR, "E:/Kiev_folder/")
# MAKE list.txt FILE
for filename in os.listdir(IMAGE_DIR):
    folder = IMAGE_DIR + filename + '/'
    f = open(folder + "list.txt", "a")
    for file in os.listdir(folder):
        if file.split(".")[1] == 'tif':
            your_path = os.path.join(folder, file)
            f.write(your_path + '\n')
    f.close()
"""

"""
# Make cmd file
IMAGE_DIR = os.path.join(ROOT_DIR, "E:/Kiev_folder/")
f = open("E:/run.cmd", "a")
for filename in os.listdir(IMAGE_DIR):
    your_path = "gdalbuildvrt.exe -resolution average -r nearest" \
                " -input_file_list E:/Kiev_folder/{}/list.txt E:/Kiev_folder/{}.vrt".format(filename, filename)
    f.write(your_path + '\n')
f.close()
"""

ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, "E:/Kiev_folder/80-0648/")
# SORT BY FOLDER
for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[1] == 'tif':
        your_path = os.path.join(IMAGE_DIR, filename)
        file_coord = filename.split(".")[0] + '.wld'
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

        img = skimage.io.imread(your_path)
        height, width, depth = img.shape
        x_min = x_coord
        y_min = y_coord - scale * height
        x_max = x_coord + scale * height
        y_max = y_coord

        for j in range (0,5):
            xj0 = x_min + j
            yJ0 = y_min + j

