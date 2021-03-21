import os
import skimage.io
from skimage.transform import rescale, resize
from PIL import Image


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

IMAGE_DIR = r'F:\car_project\ukraine\Ternopol\tif'
IMAGE_DIR_OUT = r'F:\car_project\ukraine\Ternopol\_new_256\jpg'

for filename in os.listdir(IMAGE_DIR):
    if filename.split(".")[1] == 'tif':
        print(filename)
        filename_out = filename.split(".")[0] + '.jpg'
        old_patch = os.path.join(IMAGE_DIR, filename)
        new_patch = os.path.join(IMAGE_DIR_OUT, filename_out)

        im = Image.open(old_patch)
        nx, ny = im.size

        im2 = im.resize((int(nx * 4), int(ny * 4)), Image.BICUBIC)
        # im2.save(new_patch, dpi=(288, 288))
        im2.save(new_patch, dpi=(576, 576))

        """
        im2 = im.resize((int(nx * 8), int(ny * 8)), Image.BICUBIC)
        im2.save(new_patch, dpi=(576, 576))
        """

print("OK")


