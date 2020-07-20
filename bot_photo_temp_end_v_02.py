import cv2
import os
import sys
import json
import time
import csv
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import PIL.ExifTags
import requests
from decimal import Decimal

import telebot
import logging
from telebot import types

ROOT_DIR = os.path.abspath("../")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, ""))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_roads_signs_dataset_16.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


API_TOKEN = '730715872:AAFH1dwW-C2A2j0htlxdtRQ7g-hdC9QBIxw'

bot = telebot.TeleBot(API_TOKEN)
logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 4
    # single sign detect
    # NUM_CLASSES = 1 + 17  # 1 Background + 1 Building
    NUM_CLASSES = 1 + 5  # 1 Background + 1 Building

    # detect signs by type
    # NUM_CLASSES = 1 + 35  # 1 Background + 1 Building
    # IMAGE_MAX_DIM=320
    # IMAGE_MIN_DIM=320


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()

class_names = ['BG', '2_1', '5_35_1', '5_35_2', '2.3', '2.4']

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, "Hi")

@bot.message_handler(commands=['geo'])
def geo(message):
    #chat_id = message.chat.id
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    button_geo = types.KeyboardButton(text="Send coordinate", request_location=True)
    keyboard.add(button_geo)
    bot.send_message(message.chat.id, "Press to buttom and send your coordinate ", reply_markup=keyboard)

    #bot.send_location(chat_id, '50', '30')


@bot.message_handler(content_types=['location'])
def location(message):
    if message.location is not None:
        print(message.location)
        print("latitude: %s; longitude: %s" % (message.location.latitude, message.location.longitude))


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:

        chat_id = message.chat.id

        # bot.reply_to(message, "chat_id : " + str(chat_id))

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = "D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/" + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        # bot.reply_to(message, "Photo detect")

        image = skimage.io.imread(src)
        results = model.detect([image], verbose=1)
        r = results[0]

        color_img = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        output_file_name = os.path.join("D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/photos_out/", file_info.file_path)
        skimage.io.imsave(output_file_name, color_img)

        print (r['rois'])

        i = 0
        txt = ''
        class_ids = 0
        for class_id in r['class_ids']:
            if class_id == 1:
                class_name = '2.1'
            if class_id == 2:
                class_name = '5.35.1'
            if class_id == 3:
                class_name = '5.35.2'
            if class_id == 4:
                class_name = '2.3'
            if class_id == 5:
                class_name = '2.4'
            txt = txt + " sign: " + class_name + ", score: " + str(round(r['scores'][i],2))
            class_ids = class_id
            i += 1

        # txt = "class: " + str(class_names[1]) + " score: " + str(r['scores'][0])

        # bot.reply_to(message, "Photo detect ")
        if class_ids > 0:
            bot.reply_to(message, "Photo detect: " + txt)
            bot.send_photo(chat_id=chat_id, photo=open(output_file_name, "rb"))
        else:
            bot.reply_to(message, "Photo no detect" + txt)


    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['video'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Video adding")
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['audio'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Audio adding")
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/documents/' + message.document.file_name;
        print (src)
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        image = skimage.io.imread(src)
        results = model.detect([image], verbose=1)
        r = results[0]

        color_img = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        output_file_name = os.path.join("D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/documents_out/", file_info.file_path)
        skimage.io.imsave(output_file_name, color_img)

        i = 0
        txt = ''
        class_ids = 0
        for class_id in r['class_ids']:
            if class_id == 1:
                class_name = '2.1'
            if class_id == 2:
                class_name = '5.35.1'
            else:
                class_name = '5.35.2'
            txt = txt + " sign: " + class_name + ", score: " + str(round(r['scores'][i],2))
            class_ids = class_id
            i += 1

        # txt = "class: " + str(class_names[1]) + " score: " + str(r['scores'][0])

        # bot.reply_to(message, "Photo detect ")
        if class_ids > 0:
            bot.reply_to(message, "Photo detect: " + txt)
            bot.send_photo(chat_id=chat_id, photo=open(output_file_name, "rb"))
        else:
            bot.reply_to(message, "Photo no detect" + txt)

        # file_patch = 'D:\\TEMP\\_deeplearning\\__from_kiev\\_photo_from_bot\\documents\\IMG_20190723_204955.jpg'

        img = PIL.Image.open(src)
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
        }

        # print (exif)

        GPSInfo = exif['GPSInfo']
        ExifImageWidth = exif['ExifImageWidth']
        ExifImageHeight = exif['ExifImageHeight']
        Resolution = exif['XResolution']
        FocalLength = exif['FocalLength'][0] / exif['FocalLength'][1]
        DateTimeOriginal = exif['DateTimeOriginal']
        Make = exif['Make']
        Model = exif['Model']
        Orientation = exif['Orientation']
        FocalLengthIn35mmFilm = exif['FocalLengthIn35mmFilm']
        ScaleKoef = FocalLengthIn35mmFilm / FocalLength
        Scale1 = 36 / ScaleKoef
        Scale2 = ExifImageHeight / Scale1
        
        s = r['rois'][0][2] - r['rois'][0][0]
        s2 = s/Scale2
        d = round(((FocalLength * 700 / s2)/1000),2)

        # print (GPSInfo)
        # Need check GPSInfo

        if len(GPSInfo) != 0:
            for i in GPSInfo:
                # print (i, GPSInfo[i])
                if i == 2:
                    x1 = GPSInfo[i][0][0]
                    x2 = GPSInfo[i][1][0]
                    x3 = GPSInfo[i][2][0] / GPSInfo[i][2][1]
                if i == 4:
                    y1 = GPSInfo[i][0][0]
                    y2 = GPSInfo[i][1][0]
                    y3 = GPSInfo[i][2][0] / GPSInfo[i][2][1]
                if i == 17:
                    a1 = GPSInfo[i][0] / GPSInfo[i][1]

            x = str(x1) + '.' + str(x2)
            y = str(y1) + '.' + str(y2)

            xx = Decimal(x1) + Decimal(x2) / 60 + Decimal(x3) / 3600
            yy = Decimal(y1) + Decimal(y2) / 60 + Decimal(y3) / 3600

            xx = round(xx, 10)
            yy = round(yy, 10)

            image_filename = os.path.basename(src)

            # print(xx)
            # print(yy)
            # print(a1)
            # print(FocalLength)
            # print(d)
            # print(image_filename)
            # file_patch = 'D:\\TEMP\\_deeplearning\\__from_kiev\\_photo_from_bot\\documents\\IMG_20190720_174719.jpg'
            file_patch = src
            multipart_form_data = {
                'file': (image_filename, open(file_patch, 'rb')),
                'mimType': 'image/jpeg',
                'lat': ('', str(xx)),
                'lng': ('', str(yy)),
                'dist': ('', str(d)),
                'angle': ('', str(a1)),
                'focal': ('', str(FocalLength)),
                'sign': ('', str(class_name)),
            }

            print (multipart_form_data)

            response = requests.post('http://192.168.33.80:84/php/upload.php/',
                                     files=multipart_form_data)

            print(response.status_code)

    except Exception as e:
        bot.reply_to(message, e)


# handler func=lambda
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, message.text)


bot.polling(none_stop=True, interval=0, timeout=5)