import flask
from flask import request
import requests

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

import psycopg2
from datetime import datetime

bot = telebot.TeleBot('1383386139:AAGWeMlF9BW26ZwUwnVuk2pQm6nOvUADxyw', threaded=False)

logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

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

    select_query = "SELECT number FROM avto.user WHERE chat_id = " + str(chat_id) + " LIMIT 1"

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

    select_query = "DELETE FROM avto.user WHERE chat_id = " + str(chat_id)

    cursor.execute(select_query)
    connection.commit()


stringList = {"ua": "Український", "ru": "Русский", "eng": "Englishe", "fr": "Française", "it": "Italiano",
              "de": "Deutsche", "es": "Español", "pt": "Português", "ar": "عرب", "chi": "中文", "ja": "日本人"}


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

        src = "D:/_deeplearning/__from_kiev/_photo_from_bot/" + file_info.file_path

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        # bot.send_photo(chat_id=chat_id, photo=open(src, "rb"))
        # return {"ok": True}
        # exit()
        # bot.reply_to(message, "Photo detect")

        # image = skimage.io.imread(src)

        # Detect numberplate
        # img_path = 'C:\\Users\\Administrator\\Mask_RCNN\\nomeroff-net\\examples\\images\\1.jpg'
        # img = mpimg.imread(img_path)
        img = mpimg.imread(src)
        NP = nnet.detect([img])
        # bot.reply_to(message, NP)
        # print(NP)

        # Generate image mask.
        cv_img_masks = filters.cv_img_mask(NP)

        # Detect points.
        arrPoints = rectDetector.detect(cv_img_masks)
        zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
        # bot.reply_to(message, zones)
        # print(zones)

        # find standart
        regionIds, stateIds, countLines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)
        # bot.reply_to(message, regionNames)
        # print(regionNames)

        # find text with postprocessing by standart
        textArr = textDetector.predict(zones)
        textArr = textPostprocessing(textArr, regionNames)

        # print(textArr)

        connection = connection_db()
        cursor = connection.cursor()

        for value in textArr:
            # value = translit(value, 'ru', reversed=True)
            d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
                 "T": "Т", "X": "Х"}

            value = replace_all(value, d)

            if caption is not None:
                if caption[0:1] == '@':

                    check_register_value = check_register(value, chat_id)

                    if check_register_value[0] == "0":
                        insert_query = "INSERT INTO avto.user(number, chat_id) VALUES ('" \
                                       + str(value) + "','" + str(chat_id) + "')"
                        cursor.execute(insert_query)
                        connection.commit()
                        bot.reply_to(message, "Зареєстровано з номером " + value)
                    else:
                        if len(check_register_value[1]) > 1:
                            bot.reply_to(message, check_register_value[1])
                        if len(check_register_value[2]) > 1:
                            bot.reply_to(message, check_register_value[2])
                else:
                    select_query = "SELECT number FROM avto.user WHERE chat_id = " + str(chat_id) + " LIMIT 1"
                    print(select_query)
                    cursor.execute(select_query)
                    mobile_records = cursor.fetchall()
                    # print(SQL_Query)
                    # print(mobile_records)
                    for row in mobile_records:
                        avto_number = row[0]
                        print(avto_number)

                    select_query = "SELECT chat_id FROM avto.user WHERE number = UPPER('" \
                                   + value + "') LIMIT 1"
                    print(select_query)
                    cursor.execute(select_query)
                    mobile_records = cursor.fetchall()
                    # print(SQL_Query)
                    # print(mobile_records)
                    for row in mobile_records:
                        chat_id = row[0]
                        print(chat_id)
                        # bot.reply_to(message, textQuery)

                    bot.send_photo(chat_id=chat_id, photo=open(src, "rb"))
                    bot.send_message(chat_id=chat_id, text='#' + avto_number + ' ' + caption)
                    return {"ok": True}

        cursor.close()
        # connection.close()

        # bot.reply_to(message, "Якщо номер розпізнано не вірно введіть його текстом з використанням кирилиці")

    except Exception as e:
        bot.reply_to(message, str(e))
        print(e)


# handler func=lambda
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    chat_id = message.chat.id
    print (chat_id)
    # bot.reply_to(message, message.text)

    connection = connection_db()
    cursor = connection.cursor()

    d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
         "T": "Т", "X": "Х"}

    value = replace_all(message.text.upper(), d)
    today = datetime.now()

    if value[0:1] == '@':

        check_register_value = check_register(value, chat_id)

        if check_register_value[0] == "0":
            insert_query = "INSERT INTO avto.user(number, chat_id) VALUES ('" \
                           + str(value[1:]) + "','" + str(chat_id) + "')"
            cursor.execute(insert_query)
            connection.commit()
            bot.reply_to(message, "Зареєстровано з номером " + value[1:])
        else:
            if len(check_register_value[1]) > 1:
                bot.reply_to(message, check_register_value[1])
            if len(check_register_value[2]) > 1:
                bot.reply_to(message, check_register_value[2])

        # exit()

    if value[0:1] == '#':
        try:
            select_query = "SELECT number FROM avto.user WHERE chat_id = " + str(chat_id) + " LIMIT 1"
            print(select_query)
            cursor.execute(select_query)
            mobile_records = cursor.fetchall()
            # print(SQL_Query)
            # print(mobile_records)
            for row in mobile_records:
                my_avto_number = row[0]
                print (my_avto_number)

            v = message.text.split(' ')
            avto_number_out = v[0][1:]
            mes = ' '.join(v[1:])
            print (avto_number_out)
            print (mes)

            select_query = "SELECT chat_id FROM avto.user WHERE number = UPPER('" \
                           + avto_number_out + "') LIMIT 1"
            #  + message.text.split('/')[0][1:] + "') LIMIT 1"

            print (select_query)
            cursor.execute(select_query)
            mobile_records = cursor.fetchall()
            # print(SQL_Query)
            # print(mobile_records)
            for row in mobile_records:
                chat_id_out = row[0]
                print(chat_id_out)
                # bot.reply_to(message, textQuery)
            # bot.send_message(chat_id=chat_id, text='#' + avto_number + ' # ' + message.text.split('/')[1])
            bot.send_message(chat_id=chat_id_out, text='#' + my_avto_number + ' ' + mes)
        except Exception as e:
                bot.send_message(chat_id=chat_id, text=str(e))

    if value[0:1] == '&':
        delete_registration(chat_id)
        bot.send_message(chat_id=chat_id, text='Реєстрація видалена')
    else:
        bot.send_message(chat_id=chat_id, text=message.text)
        # exit()


@app.route("/", methods=["GET", "POST"])
def receive_update():
    if request.method == "POST":
        if flask.request.headers.get('content-type') == 'application/json':
            json_string = flask.request.get_data().decode('utf-8')
            update = telebot.types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return ''
        else:
            flask.abort(403)
    return {"ok": True}


if __name__ == '__main__':
    #app.run()
    app.run(host='127.0.0.1', port=5000, debug=False)

