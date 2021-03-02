import os
import numpy as np
import skimage.io
from matplotlib import pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

array = ['jpg']

for item in array:
    # IMAGE_DIR = os.path.join(ROOT_DIR, "F:/Kiev_tile/80-0734/")
    # IMAGE_DIR = 'F:/Kiev_tile/' + item + '/'
    IMAGE_DIR = 'F:/Poland/varshava_image_in/'

    for filename in os.listdir(IMAGE_DIR):
        if filename.split(".")[1] == 'jpg':
            path = os.path.join(IMAGE_DIR, filename.split(".")[0])
            os.mkdir(path)
            print (filename)
            your_path = os.path.join(IMAGE_DIR, filename)
            file_coord = filename.split(".")[0] + '.jpgw'
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
            # image_all = np.flipud(image_all)

            print(filename, height, width, depth, size, scale, x_coord, y_coord)

            for i in range(0, height-256, 256):
                y_coord_new = y_coord - i * scale
                for j in range(0, width-256, 256):
                    cropped = image_all[i:i+256, j:j+256]
                    x_coord_new = x_coord + j * scale
                    skimage.io.imshow(cropped)
                    plt.show()
                    f = open(os.path.join(path, 'f' + '_' + str(i) + '_' + str(j) + '.jgw'), "w+")
                    f.write(str(scale) + '\r\n')
                    f.write(str("0") + '\r\n')
                    f.write(str("0") + '\r\n')
                    f.write("-" + str(scale) + '\r\n')
                    f.write(str(x_coord_new) + '\r\n')
                    f.write(str(y_coord_new) + '\r\n')
                    f.close()
                    skimage.io.imsave(os.path.join(path, 'f' + '_' + str(i) + '_' + str(j) + '.jpg'), cropped)

            print("")

print("End")
