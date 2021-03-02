import flask
from flask import Flask, request, Response
from viberbot import Api
from viberbot.api.bot_configuration import BotConfiguration
from viberbot.api.messages import VideoMessage
from viberbot.api.messages.text_message import TextMessage
import json

from viberbot.api.viber_requests import ViberConversationStartedRequest
from viberbot.api.viber_requests import ViberFailedRequest
from viberbot.api.viber_requests import ViberMessageRequest
from viberbot.api.viber_requests import ViberSubscribedRequest
from viberbot.api.viber_requests import ViberUnsubscribedRequest

import requests

from flask import send_file

from viberbot.api.messages import (
    TextMessage,
    ContactMessage,
    PictureMessage,
    VideoMessage
)

from viberbot.api.messages.data_types.contact import Contact

import psycopg2

import os
import numpy as np
import sys
import matplotlib.image as mpimg

import skimage.io

import telebot
import time
import ast
import logging
from telebot import types

from datetime import datetime
import urllib.request

patch_photo_telegram = "D:/_deeplearning/__from_kiev/_photo_from_bot/"
patch_photo_viber = "D:/_deeplearning/__from_kiev/_photo_from_bot/photos/viber/"

d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І",
     "K": "К", "M": "М", "O": "О", "P": "Р","T": "Т", "X": "Х"}

stringList = {"ua": "Український", "ru": "Русский", "eng": "Englishe", "fr": "Française", "it": "Italiano",
              "de": "Deutsche", "es": "Español", "pt": "Português", "ar": "عرب", "chi": "中文", "ja": "日本人"}

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
# MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_DIR = NOMEROFF_NET_DIR
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, \
    textPostprocessingAsync

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")

viber = Api(BotConfiguration(
    name='NumberCarBot',
    avatar='',
    auth_token='4bdc538a20e7d1b3-21905a4be00e2fd3-e79c68a1197c6ec8'
))

bot = telebot.TeleBot('922151292:AAGAMicVQit6cRf94SZ3P50gTZhXNQMYRZg', threaded=False)

logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

app = flask.Flask(__name__)


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))


def connection_db():
    connection = psycopg2.connect(user="postgres",
                                  password="postgres",
                                  host="192.168.33.89",
                                  port="5432",
                                  database="geo_new_3")
    # cursor = connection.cursor()
    return connection


def check_register(number, chat_id):
    number_msg = check_register_car_number(number)
    chat_id_msg = check_register_chat_id(chat_id)

    if len(number_msg) > 1 or len(chat_id_msg) > 1:
        return ["1", number_msg, chat_id_msg]
    else:
        return ["0"]


def check_register_car_number(number):
    if number[0] == "@":
        car_number = number[1:]
    else:
        car_number = number

    connection = connection_db()
    cursor = connection.cursor()

    select_query = "SELECT * FROM avto.user WHERE number = '" \
                   + car_number + "' LIMIT 1"
    print(select_query)

    cursor.execute(select_query)

    data = cursor.fetchone()

    if data is not None:
        number_msg = 'Номер ' + str(car_number) + ' вже зареєстровано'
    else:
        number_msg = ''

    return number_msg


def check_register_chat_id(chat_id):
    connection = connection_db()
    cursor = connection.cursor()

    select_query = "SELECT number FROM avto.user WHERE chat_id = '" + str(chat_id) + "' LIMIT 1"

    print(select_query)

    cursor.execute(select_query)

    data = cursor.fetchone()

    if data is not None:
        chat_id_msg = 'Ви вже зареєстровані в системі з номером ' + str(data[0])
    else:
        chat_id_msg = ''

    return chat_id_msg


def delete_registration(chat_id):
    connection = connection_db()
    cursor = connection.cursor()

    select_query = "DELETE FROM avto.user WHERE chat_id = '" + str(chat_id) + "'"

    cursor.execute(select_query)
    connection.commit()


def makeKeyboard():
    markup = types.InlineKeyboardMarkup()

    for key, value in stringList.items():
        markup.add(types.InlineKeyboardButton(text=value,
                                              callback_data="['value', '" + value + "', '" + key + "']"))

    return markup


def returnHelp(leng, chat_id):
    print(leng)
    connection = connection_db()
    cursor = connection.cursor()

    select_query = "SELECT value FROM avto.about_info WHERE key = '" \
                   + leng + "' LIMIT 1"

    cursor.execute(select_query)
    mobile_records = cursor.fetchall()
    # print(SQL_Query)
    # print(mobile_records)
    for row in mobile_records:
        help_text = row[0]
        help_text = help_text.strip()

    # return help_text
    bot.send_message(chat_id=chat_id, text=help_text[0:1000])


def select_chat_and_number(chat_id, avto_number_out):
    connection = connection_db()
    cursor = connection.cursor()

    out = []

    select_query = "SELECT number FROM avto.user WHERE chat_id = '" + str(chat_id) + "' LIMIT 1"
    print(select_query)
    cursor.execute(select_query)
    mobile_records = cursor.fetchall()
    # print(SQL_Query)
    # print(mobile_records)
    for row in mobile_records:
        my_avto_number = row[0]
        out.append(my_avto_number)
        print("my_avto_number = " + my_avto_number)

    select_query = "SELECT chat_id, messenger FROM avto.user WHERE number = UPPER('" \
                   + avto_number_out + "') LIMIT 1"
    #  + message.text.split('/')[0][1:] + "') LIMIT 1"

    print(select_query)
    cursor.execute(select_query)
    mobile_records = cursor.fetchall()
    # print(SQL_Query)
    # print(mobile_records)
    for row in mobile_records:
        chat_id_out = row[0]
        out.append(chat_id_out)
        messenger = row[1]
        out.append(messenger)
        print(chat_id_out)

    return (out)


def registration(value, chat_id, messenger_type):
    connection = connection_db()
    cursor = connection.cursor()

    insert_query = "INSERT INTO avto.user(number, chat_id) " \
                   "VALUES ('" + str(value) + "','" + str(chat_id) + "','" + str(messenger_type) + "')"

    cursor.execute(insert_query)
    connection.commit()


def number_ocr(src):
    img = mpimg.imread(src)
    NP = nnet.detect([img])

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

    for value in textArr:
        value = replace_all(value, d)

    return value


def replace_value(value):
    value = replace_all(value, d)
    return value


@bot.message_handler(commands=['help', 'start'])
def handle_command_adminwindow(message):
    bot.send_message(chat_id=message.chat.id,
                     text="Choose language",
                     reply_markup=makeKeyboard(),
                     parse_mode='HTML')


@bot.callback_query_handler(func=lambda call: True)
def handle_query(call):

    chat_id = call.from_user.id

    if (call.data.startswith("['value'")):
        print(f"call.data : {call.data} , type : {type(call.data)}")
        print(f"ast.literal_eval(call.data) : {ast.literal_eval(call.data)} , type : {type(ast.literal_eval(call.data))}")
        valueFromCallBack = ast.literal_eval(call.data)[1]
        keyFromCallBack = ast.literal_eval(call.data)[2]
        returnHelp(keyFromCallBack, chat_id)
        # bot.answer_callback_query(callback_query_id=call.id,
                              # show_alert=True,
                              # text=returnHelp(keyFromCallBack))
                              #text="You Clicked " + valueFromCallBack + " and key is " + keyFromCallBack)

    if (call.data.startswith("['key'")):
        keyFromCallBack = ast.literal_eval(call.data)[1]
        del stringList[keyFromCallBack]
        bot.edit_message_text(chat_id=call.message.chat.id,
                              text="Here are the values of stringList",
                              message_id=call.message.message_id,
                              reply_markup=makeKeyboard(),
                              parse_mode='HTML')


@bot.message_handler(commands=['register'])
def send_welcome(message):
    bot.reply_to(message, "Для реєстрації надішліть номер свого авто")


@bot.message_handler(commands=['geo'])
def geo(message):
    # chat_id = message.chat.id
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    button_geo = types.KeyboardButton(text="Send coordinate", request_location=True)
    keyboard.add(button_geo)
    bot.send_message(message.chat.id, "Press to buttom and send your coordinate ", reply_markup=keyboard)

    # bot.send_location(chat_id, '50', '30')


@bot.message_handler(content_types=['location'])
def location(message):
    if message.location is not None:
        print(message.location)
        print("latitude: %s; longitude: %s" % (message.location.latitude, message.location.longitude))


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id
        caption = message.caption

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = patch_photo_telegram + file_info.file_path

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        value = number_ocr(src)
        print("распознанный номер = " + str(value))

        if caption is not None:
            if caption[0:1] == '@':

                check_register_value = check_register(value, chat_id)

                if check_register_value[0] == "0":
                    registration(value, chat_id, 't')
                    bot.reply_to(message, "Зареєстровано з номером " + value)
                else:
                    if len(check_register_value[1]) > 1:
                        bot.reply_to(message, check_register_value[1])
                    if len(check_register_value[2]) > 1:
                        bot.reply_to(message, check_register_value[2])
            else:

                my_avto_number, chat_id_out, messenger = select_chat_and_number(chat_id, value)

                if messenger == 't':
                    bot.send_photo(chat_id=chat_id_out, photo=open(src, "rb"))
                    bot.send_message(chat_id=chat_id_out, text='#' + my_avto_number + ' ' + caption)
                if messenger == 'v':
                    # viber.send_messages(chat_id_out, [
                    #     PictureMessage(text='#' + my_avto_number + ' ' + caption, media=open(src, "rb"))
                    # ])
                    viber.send_messages(chat_id_out, [
                        TextMessage(text='#' + my_avto_number + ' ' + caption)
                    ])

                return {"ok": True}

    except Exception as e:
        bot.reply_to(message, str(e))
        print(e)


# handler func=lambda
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    try:
        chat_id = message.chat.id
        value = replace_all(message.text.upper(), d)

        if value[0:1] == '@':

            check_register_value = check_register(value, chat_id)

            if check_register_value[0] == "0":
                registration(value[1:], chat_id, 't')
                bot.reply_to(message, "Зареєстровано з номером " + value[1:])
            else:
                if len(check_register_value[1]) > 1:
                    bot.reply_to(message, check_register_value[1])
                if len(check_register_value[2]) > 1:
                    bot.reply_to(message, check_register_value[2])

        if value[0:1] == '#':

            v = message.text.split(' ')
            avto_number_out = v[0][1:]
            mes = ' '.join(v[1:])
            print("avto_number_out " + avto_number_out)
            print("message" + mes)

            my_avto_number, chat_id_out, messenger = select_chat_and_number(chat_id, avto_number_out)

            if messenger == 't':
                bot.send_message(chat_id=chat_id_out, text='#' + my_avto_number + ' ' + mes)
            if messenger == 'v':
                viber.send_messages(chat_id_out, [
                    TextMessage(text='#' + my_avto_number + ' ' + mes)
                ])

        if value[0:1] == '&':
            delete_registration(chat_id)
            bot.send_message(chat_id=chat_id, text='Реєстрація видалена')

    except Exception as e:
        bot.send_message(chat_id=chat_id, text=str(e))


@app.route('/', methods=['POST'])
def receive_update():
    if request.method == "POST":
        if 'update_id' in request.json:
            try:
                # telegram
                print('telegram ok')
                if flask.request.headers.get('content-type') == 'application/json':
                    json_string = flask.request.get_data().decode('utf-8')
                    update = telebot.types.Update.de_json(json_string)
                    bot.process_new_updates([update])
                    return ''
                else:
                    flask.abort(403)
                return {"ok": True}
            except Exception as e:
                send_message(chat_id, text=str(e))
        else:
            # viber
            print('viber ok')
            logging.debug("received request. post data: {0}".format(request.get_data()))
            # every viber message is signed, you can verify the signature using this method
            if not viber.verify_signature(request.get_data(), request.headers.get('X-Viber-Content-Signature')):
                return Response(status=403)

            # this library supplies a simple way to receive a request object
            viber_request = viber.parse_request(request.get_data())

            if isinstance(viber_request, ViberMessageRequest):
                message = viber_request.message
                request_data = request.get_data()
                request_dict = json.loads(request_data)
                t = request_dict["message"]["type"]
                if t == 'text':
                    print('send text')
                    value = replace_all(message.text.upper(), d)
                    chat_id = viber_request.sender.id
                    try:
                        if value[0:1] == '@':
                            check_register_value = check_register(value, chat_id)

                            if check_register_value[0] == "0":
                                registration(value[1:], chat_id, 'v')
                                viber.send_messages(viber_request.sender.id, [
                                    TextMessage(text="Зареєстровано з номером " + value[1:])
                                ])
                            else:
                                print(check_register_value[1])
                                print(check_register_value[2])
                                if len(check_register_value[1]) > 1:
                                    viber.send_messages(viber_request.sender.id, [
                                        TextMessage(text=check_register_value[1])
                                        # check_register_value[1]
                                    ])
                                if len(check_register_value[2]) > 1:
                                    viber.send_messages(viber_request.sender.id, [
                                        TextMessage(text=check_register_value[2])
                                        # check_register_value[2]
                                    ])

                        if value[0:1] == '#':
                            v = message.text.split(' ')
                            avto_number_out = v[0][1:]
                            mes = ' '.join(v[1:])
                            print("avto_number_out " + avto_number_out)
                            print("message" + mes)

                            my_avto_number, chat_id_out, messenger = select_chat_and_number(chat_id, avto_number_out)

                            if messenger == 't':
                                bot.send_message(chat_id=chat_id_out, text='#' + my_avto_number + ' ' + mes)
                            if messenger == 'v':
                                viber.send_messages(chat_id_out, [
                                    TextMessage(text='#' + my_avto_number + ' ' + mes)
                                ])

                        if value[0:1] == '&':
                            delete_registration(chat_id)
                            viber.send_messages(viber_request.sender.id, [
                                TextMessage(text='Реєстрація видалена')
                            ])

                    except Exception as e:
                        viber.send_messages(viber_request.sender.id, [
                            TextMessage(text=str(e))
                        ])

                if t == 'picture':
                    chat_id = viber_request.sender.id
                    caption = message.text
                    file_name = str(viber_request.sender.id) + str(time.time()) + str(".jpg")
                    src = patch_photo_viber + file_name
                    urllib.request.urlretrieve(message.media, src)
                    value = number_ocr(src)
                    print("распознанный номер = " + str(value))
                    try:
                        if caption is not None:
                            if caption[0:1] == '@':

                                check_register_value = check_register(value, chat_id)

                                if check_register_value[0] == "0":
                                    registration(value, chat_id, 'v')
                                    viber.send_messages(viber_request.sender.id, [
                                        TextMessage(text="Зареєстровано з номером " + value)
                                    ])
                                else:
                                    if len(check_register_value[1]) > 1:
                                        viber.send_messages(viber_request.sender.id, [
                                            TextMessage(text=check_register_value[1])
                                            # check_register_value[1]
                                        ])
                                    if len(check_register_value[2]) > 1:
                                        viber.send_messages(viber_request.sender.id, [
                                            TextMessage(text=check_register_value[2])
                                            # check_register_value[2]
                                        ])
                            else:

                                my_avto_number, chat_id_out, messenger = select_chat_and_number(chat_id, value)

                                if messenger == 't':
                                    bot.send_photo(chat_id=chat_id_out, photo=open(src, "rb"))
                                    bot.send_message(chat_id=chat_id_out,
                                                     text='#' + str(my_avto_number) + ' ' + caption)
                                if messenger == 'v':
                                    viber.send_messages(chat_id_out, [
                                        PictureMessage(text='#' + my_avto_number + ' ' + caption,
                                                       media=message.media)
                                    ])
                                    viber.send_messages(chat_id_out, [
                                        TextMessage(text='#' + my_avto_number + ' ' + caption)
                                    ])

                    except Exception as e:
                        viber.send_messages(viber_request.sender.id, [
                            TextMessage(text=str(e))
                        ])

            elif isinstance(viber_request, ViberSubscribedRequest):
                viber.send_messages(viber_request.get_user.id, [
                    TextMessage(text="thanks for subscribing!")
                ])
            elif isinstance(viber_request, ViberConversationStartedRequest):
                viber.send_messages(viber_request.get_user().get_id(), [
                    TextMessage(text="Welcome!")
                ])
            elif isinstance(viber_request, ViberFailedRequest):
                logging.warn("client failed receiving message. failure: {0}".format(viber_request))

            return Response(status=200)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
