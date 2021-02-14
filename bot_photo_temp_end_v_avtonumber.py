import os
import numpy as np
import sys
import matplotlib.image as mpimg

import skimage.io

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
# MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_DIR = NOMEROFF_NET_DIR
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

IMAGE_DIR = 'F:/car_video/image/'

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")

f = open("F:/car_video/out.txt", "a")

# Detect numberplate


for filename in os.listdir(IMAGE_DIR):

    f_id = filename[0:7]
    if filename.split(".")[1] == 'jpg':

        try:
            your_path = os.path.join(IMAGE_DIR, filename)

            # image_all = skimage.io.imread(your_path)
            # height, width, depth = image_all.shape

            size = os.path.getsize(your_path)

            # your_path = 'F:/car_video/image/id_0001_f_0023.jpg'
            img = mpimg.imread(your_path)
            NP = nnet.detect([img])

            # Generate image mask.
            cv_img_masks = filters.cv_img_mask(NP)

            # Detect points.
            arrPoints = rectDetector.detect(cv_img_masks)
            zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

            # find standart
            regionIds, stateIds, countLines = optionsDetector.predict(zones)
            regionNames = optionsDetector.getRegionLabels(regionIds)

            # find text with postprocessing by standart
            textArr = textDetector.predict(zones)
            textArr = textPostprocessing(textArr, regionNames)
            if len(textArr) == 0:
                print(f_id, ' ', filename, ' not_number', ' ', str(size))
                f.write(f_id + ' ' + filename + ' not_number' + ' ' + str(size) + '\n')
            else:
                print(f_id, ' ', filename, ' ', textArr[0], ' ', str(size))
                f.write(f_id + ' ' + filename + ' ' + textArr[0] + ' ' + str(size) + '\n')

        except Exception as e:
            print(e)

f.close()

print("End processing")
