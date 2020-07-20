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

import telebot
import logging
from telebot import types

ROOT_DIR = os.path.abspath("../")

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, ""))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_road_signs_0026.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# API_TOKEN = '730715872:AAFH1dwW-C2A2j0htlxdtRQ7g-hdC9QBIxw'
API_TOKEN = '1243206219:AAHgio7bYbEvkr2JNwNwCTNFDNOutI1c8Mg'

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
    # NUM_CLASSES = 1 + 1  # 1 Background + 1 Building

    # detect signs by type
    NUM_CLASSES = 1 + 168  # 1 Background + 1 Building
    # IMAGE_MAX_DIM=320
    # IMAGE_MIN_DIM=320


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()

# class_names = ['BG', 'road_signs']

class_names = ['BG', '1.1', '1.11', '1.12', '1.13', '1.19', '1.2', '1.20', '1.22', '1.23.1', '1.24', '1.25', '1.26',
               '1.27', '1.28', '1.29', '1.30', '1.3.1', '1.32', '1.33', '1.34', '1.37', '1.39', '1.4.1', '1.4.2',
               '1.4.3', '1.5.2', '1.5.3', '1.6', '1.7', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '3.1', '3.12',
               '3.14', '3.15', '3.18', '3.2', '3.21', '3.22', '3.23', '3.24', '3.27', '3.29', '3.3', '3.30', '3.31',
               '3.32', '3.33', '3.34', '3.35', '3.36', '3.37', '3.38', '3.39', '3.4', '3.41', '3.42', '3.9', '4.1',
               '4.10', '4.11', '4.12', '4.13', '4.14', '4.2', '4.21', '4.22', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8',
               '4.9', '5.10.1', '5.10.2', '5.10.3', '5.11', '5.12', '5.16', '5.17.1', '5.17.2', '5.18', '5.19',
               '5.20.1', '5.20.3', '5.21.1', '5.21.2', '5.26', '5.27', '5.29.1', '5.29.2', '5.29.3', '5.30', '5.31',
               '5.32', '5.33', '5.35.1', '5.35.2', '5.36.1', '5.36.2', '5.37.1', '5.37.2', '5.38', '5.39', '5.40',
               '5.41.1', '5.41.2', '5.42.1', '5.42.2', '5.43.1', '5.43.2', '5.5', '5.54', '5.6', '5.60', '5.62',
               '5.64', '5.70', '5.7.1', '5.7.2', '6.1', '6.11', '6.16', '6.5', '6.6', '6.7.1', '6.7.2', '6.8',
               '7.1.1', '7.12', '7.13', '7.1.3', '7.14', '7.1.4', '7.16', '7.17', '7.18', '7.2.1', '7.2.2', '7.2.3',
               '7.2.4', '7.2.5', '7.2.6', '7.3.1', '7.3.2', '7.3.3', '7.4.1', '7.4.4', '7.4.6', '7.4.7', '7.5.1',
               '7.5.4', '7.5.6', '7.5.7', '7.6.1', '7.6.2', '7.6.3', '7.6.4', '7.6.5', '7.6.6', '7.8', '7.9', 'TIP']

def random_colors(N):
    np.random.seed(1)
    # colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    colors = [(234, 237, 58) for _ in range(N)]
    # print (colors)
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
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (43, 6, 6), 2
        )

    return image


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Привет! 
Я тестовый бот, и умею я пока немного. Напиши мне и я отвечу!\
""")

@bot.message_handler(commands=['geo'])
def geo(message):
    #chat_id = message.chat.id
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    button_geo = types.KeyboardButton(text="Отправить местоположение", request_location=True)
    keyboard.add(button_geo)
    bot.send_message(message.chat.id, "Нажми на кнопку и передай свое местоположение", reply_markup=keyboard)

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

        src = "D:/_deeplearning/__from_kiev/_photo_from_bot/" + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        # bot.reply_to(message, "Фото добавлено")

        image = skimage.io.imread(src)
        results = model.detect([image], verbose=1)
        r = results[0]

        color_img = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        output_file_name = os.path.join("D:/_deeplearning/__from_kiev/_photo_from_bot/photos_out/", file_info.file_path)
        skimage.io.imsave(output_file_name, color_img)

        print (r['rois'])

        bot.send_photo(chat_id=chat_id, photo=open(output_file_name, "rb"))

        """
        i = 0
        txt = ''
        class_ids = 0
        for class_id in r['class_ids']:
            if class_id == 1:
                class_name = 'road_signs'
            txt = class_name
            class_ids = class_id
            i += 1

        txt = "class: " + str(class_names[1]) + " score: " + str(r['scores'][0])

        # bot.reply_to(message, "Фото распознано ")
        if class_ids > 0:
            bot.reply_to(message, "Фото розпізнано: " + txt)
            bot.send_photo(chat_id=chat_id, photo=open(output_file_name, "rb"))
        else:
            bot.reply_to(message, "Фото не розпізнано. Спробуйте змінити освітлення або наблизити знак." + txt)
        """

    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['video'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'E:/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Видео добавлено")
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['audio'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'E:/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Аудио добавлено")
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'E:/_deeplearning/__from_kiev/_photo_from_bot/documents/' + message.document.file_name;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Файл добавлено")
    except Exception as e:
        bot.reply_to(message, e)


# handler func=lambda должна быть ниже handler commands. Иначе начинает работать первой
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, message.text)


bot.polling(none_stop=True, interval=0, timeout=5)