"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_road_signs_0030_2020_03_25.h5")

# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_p1.h5")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_road_signs_0016.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "road_signs"

    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 168  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5000

    VALIDATION_STEPS = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.002


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        # self.add_class("road_signs", 1, "road_sign")
        # self.add_class("road_signs", 2, "error")
        # self.add_class("road_signs", 3, "3")
        # self.add_class("road_signs", 4, "4")

        self.add_class("road_signs", 1, "1.1")
        self.add_class("road_signs", 2, "1.11")
        self.add_class("road_signs", 3, "1.12")
        self.add_class("road_signs", 4, "1.13")
        self.add_class("road_signs", 5, "1.19")
        self.add_class("road_signs", 6, "1.2")
        self.add_class("road_signs", 7, "1.20")
        self.add_class("road_signs", 8, "1.22")
        self.add_class("road_signs", 9, "1.23.1")
        self.add_class("road_signs", 10, "1.24")
        self.add_class("road_signs", 11, "1.25")
        self.add_class("road_signs", 12, "1.26")
        self.add_class("road_signs", 13, "1.27")
        self.add_class("road_signs", 14, "1.28")
        self.add_class("road_signs", 15, "1.29")
        self.add_class("road_signs", 16, "1.30")
        self.add_class("road_signs", 17, "1.3.1")
        self.add_class("road_signs", 18, "1.32")
        self.add_class("road_signs", 19, "1.33")
        self.add_class("road_signs", 20, "1.34")
        self.add_class("road_signs", 21, "1.37")
        self.add_class("road_signs", 22, "1.39")
        self.add_class("road_signs", 23, "1.4.1")
        self.add_class("road_signs", 24, "1.4.2")
        self.add_class("road_signs", 25, "1.4.3")
        self.add_class("road_signs", 26, "1.5.2")
        self.add_class("road_signs", 27, "1.5.3")
        self.add_class("road_signs", 28, "1.6")
        self.add_class("road_signs", 29, "1.7")
        self.add_class("road_signs", 30, "2.1")
        self.add_class("road_signs", 31, "2.2")
        self.add_class("road_signs", 32, "2.3")
        self.add_class("road_signs", 33, "2.4")
        self.add_class("road_signs", 34, "2.5")
        self.add_class("road_signs", 35, "2.6")
        self.add_class("road_signs", 36, "3.1")
        self.add_class("road_signs", 37, "3.12")
        self.add_class("road_signs", 38, "3.14")
        self.add_class("road_signs", 39, "3.15")
        self.add_class("road_signs", 40, "3.18")
        self.add_class("road_signs", 41, "3.2")
        self.add_class("road_signs", 42, "3.21")
        self.add_class("road_signs", 43, "3.22")
        self.add_class("road_signs", 44, "3.23")
        self.add_class("road_signs", 45, "3.24")
        self.add_class("road_signs", 46, "3.27")
        self.add_class("road_signs", 47, "3.29")
        self.add_class("road_signs", 48, "3.3")
        self.add_class("road_signs", 49, "3.30")
        self.add_class("road_signs", 50, "3.31")
        self.add_class("road_signs", 51, "3.32")
        self.add_class("road_signs", 52, "3.33")
        self.add_class("road_signs", 53, "3.34")
        self.add_class("road_signs", 54, "3.35")
        self.add_class("road_signs", 55, "3.36")
        self.add_class("road_signs", 56, "3.37")
        self.add_class("road_signs", 57, "3.38")
        self.add_class("road_signs", 58, "3.39")
        self.add_class("road_signs", 59, "3.4")
        self.add_class("road_signs", 60, "3.41")
        self.add_class("road_signs", 61, "3.42")
        self.add_class("road_signs", 62, "3.9")
        self.add_class("road_signs", 63, "4.1")
        self.add_class("road_signs", 64, "4.10")
        self.add_class("road_signs", 65, "4.11")
        self.add_class("road_signs", 66, "4.12")
        self.add_class("road_signs", 67, "4.13")
        self.add_class("road_signs", 68, "4.14")
        self.add_class("road_signs", 69, "4.2")
        self.add_class("road_signs", 70, "4.21")
        self.add_class("road_signs", 71, "4.22")
        self.add_class("road_signs", 72, "4.3")
        self.add_class("road_signs", 73, "4.4")
        self.add_class("road_signs", 74, "4.5")
        self.add_class("road_signs", 75, "4.6")
        self.add_class("road_signs", 76, "4.7")
        self.add_class("road_signs", 77, "4.8")
        self.add_class("road_signs", 78, "4.9")
        self.add_class("road_signs", 79, "5.10.1")
        self.add_class("road_signs", 80, "5.10.2")
        self.add_class("road_signs", 81, "5.10.3")
        self.add_class("road_signs", 82, "5.11")
        self.add_class("road_signs", 83, "5.12")
        self.add_class("road_signs", 84, "5.16")
        self.add_class("road_signs", 85, "5.17.1")
        self.add_class("road_signs", 86, "5.17.2")
        self.add_class("road_signs", 87, "5.18")
        self.add_class("road_signs", 88, "5.19")
        self.add_class("road_signs", 89, "5.20.1")
        self.add_class("road_signs", 90, "5.20.3")
        self.add_class("road_signs", 91, "5.21.1")
        self.add_class("road_signs", 92, "5.21.2")
        self.add_class("road_signs", 93, "5.26")
        self.add_class("road_signs", 94, "5.27")
        self.add_class("road_signs", 95, "5.29.1")
        self.add_class("road_signs", 96, "5.29.2")
        self.add_class("road_signs", 97, "5.29.3")
        self.add_class("road_signs", 98, "5.30")
        self.add_class("road_signs", 99, "5.31")
        self.add_class("road_signs", 100, "5.32")
        self.add_class("road_signs", 101, "5.33")
        self.add_class("road_signs", 102, "5.35.1")
        self.add_class("road_signs", 103, "5.35.2")
        self.add_class("road_signs", 104, "5.36.1")
        self.add_class("road_signs", 105, "5.36.2")
        self.add_class("road_signs", 106, "5.37.1")
        self.add_class("road_signs", 107, "5.37.2")
        self.add_class("road_signs", 108, "5.38")
        self.add_class("road_signs", 109, "5.39")
        self.add_class("road_signs", 110, "5.40")
        self.add_class("road_signs", 111, "5.41.1")
        self.add_class("road_signs", 112, "5.41.2")
        self.add_class("road_signs", 113, "5.42.1")
        self.add_class("road_signs", 114, "5.42.2")
        self.add_class("road_signs", 115, "5.43.1")
        self.add_class("road_signs", 116, "5.43.2")
        self.add_class("road_signs", 117, "5.5")
        self.add_class("road_signs", 118, "5.54")
        self.add_class("road_signs", 119, "5.6")
        self.add_class("road_signs", 120, "5.60")
        self.add_class("road_signs", 121, "5.62")
        self.add_class("road_signs", 122, "5.64")
        self.add_class("road_signs", 123, "5.70")
        self.add_class("road_signs", 124, "5.7.1")
        self.add_class("road_signs", 125, "5.7.2")
        self.add_class("road_signs", 126, "6.1")
        self.add_class("road_signs", 127, "6.11")
        self.add_class("road_signs", 128, "6.16")
        self.add_class("road_signs", 129, "6.5")
        self.add_class("road_signs", 130, "6.6")
        self.add_class("road_signs", 131, "6.7.1")
        self.add_class("road_signs", 132, "6.7.2")
        self.add_class("road_signs", 133, "6.8")
        self.add_class("road_signs", 134, "7.1.1")
        self.add_class("road_signs", 135, "7.12")
        self.add_class("road_signs", 136, "7.13")
        self.add_class("road_signs", 137, "7.1.3")
        self.add_class("road_signs", 138, "7.14")
        self.add_class("road_signs", 139, "7.1.4")
        self.add_class("road_signs", 140, "7.16")
        self.add_class("road_signs", 141, "7.17")
        self.add_class("road_signs", 142, "7.18")
        self.add_class("road_signs", 143, "7.2.1")
        self.add_class("road_signs", 144, "7.2.2")
        self.add_class("road_signs", 145, "7.2.3")
        self.add_class("road_signs", 146, "7.2.4")
        self.add_class("road_signs", 147, "7.2.5")
        self.add_class("road_signs", 148, "7.2.6")
        self.add_class("road_signs", 149, "7.3.1")
        self.add_class("road_signs", 150, "7.3.2")
        self.add_class("road_signs", 151, "7.3.3")
        self.add_class("road_signs", 152, "7.4.1")
        self.add_class("road_signs", 153, "7.4.4")
        self.add_class("road_signs", 154, "7.4.6")
        self.add_class("road_signs", 155, "7.4.7")
        self.add_class("road_signs", 156, "7.5.1")
        self.add_class("road_signs", 157, "7.5.4")
        self.add_class("road_signs", 158, "7.5.6")
        self.add_class("road_signs", 159, "7.5.7")
        self.add_class("road_signs", 160, "7.6.1")
        self.add_class("road_signs", 161, "7.6.2")
        self.add_class("road_signs", 162, "7.6.3")
        self.add_class("road_signs", 163, "7.6.4")
        self.add_class("road_signs", 164, "7.6.5")
        self.add_class("road_signs", 165, "7.6.6")
        self.add_class("road_signs", 166, "7.8")
        self.add_class("road_signs", 167, "7.9")
        self.add_class("road_signs", 168, "TIP")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            class_ids = [int(n['class']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "road_signs",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "road_signs":
            # return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        #class_ids = image_info['class']

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        #mask = np.ones([info["height"], info["width"], len(info["polygons"])],
        #                dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        # if info["source"] == "road_signs":
            # return info["path"]
        # else:
            # super(self.__class__, self).image_reference(image_id)
        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    """
    augmentation = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-2, 2))
        ], random_order=True) # apply augmenters in random order
    """

    """
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    augmentation = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            iaa.Crop(percent=(0, 0.1)),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-25, 25), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255) # if mode is constant, use a cval between 0 and 255
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    """
    augmentation = iaa.OneOf([
        iaa.Affine(scale=(0.5, 1)),
        # iaa.Affine(scale=(0.75, 1.25)), 
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Affine(rotate=(-10, 10)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    print("Training all layers")
    model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=30,
                augmentation=augmentation,
                # epochs=40,
                # layers='heads')
                # layers='4+')
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
