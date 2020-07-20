import os
import numpy as np
import sys
import matplotlib.image as mpimg

import skimage.io

import telebot
import logging
from telebot import types

import psycopg2
from datetime import datetime

import cv2

# API_TOKEN = '730715872:AAFH1dwW-C2A2j0htlxdtRQ7g-hdC9QBIxw' #ngo_chat_bot

# NumberCarBot
API_TOKEN = '922151292:AAGAMicVQit6cRf94SZ3P50gTZhXNQMYRZg'

bot = telebot.TeleBot(API_TOKEN)
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

@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    description = '  Бот призначений для спілкування учасників дорожнього руху. ' \
                  'Спілкування між учасниками проходить по номеру транспортного засобу. ' \
                  '  Для користування ботом, спочатку необхідно зареєструватись. ' \
                  'Для цього можно просто сфотографувати свій номер транспортного засобу ' \
                  'і надіслати фото в бот з підписом "@". Або можно відправити інформацію у форматі @номер . ' \
                  'Де "номер" - це номер вашого трансортного засобу. ' \
                  '  Для відправки повідомлення користувачу необхідно просто сфотографувати ' \
                  'транспортний засіб користувача з номером і в описі фото ввести повідомлення. ' \
                  'Або можно відправити повідомлення у форматі - #номер повідомлення . ' \
                  'Де "номер" це номер транспортного засоба, якому ви хочете відправити повідомлення, ' \
                  '"повідомлення" - це текст повідомлення. ' \
                  'Тобто, спочатку треба ввести сімвол # потім номер адресата ' \
                  'потім пробел а потім саме повідомлення. І відпрвити. ' \
                  'Приємного користування.'
    bot.reply_to(message, description)


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

        # bot.reply_to(message, "chat_id : " + str(chat_id))

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = "D:/_deeplearning/__from_kiev/_photo_from_bot/" + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
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

        connection = psycopg2.connect(user = "postgres",
                                      password = "postgres",
                                      host = "192.168.33.89",
                                      port = "5432",
                                      database = "geo_new_3")
        cursor = connection.cursor()

        for value in textArr:
            # value = translit(value, 'ru', reversed=True)
            d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
                 "T": "Т", "X": "Х"}

            value = replace_all(value, d)
            today = datetime.now()

            insert_query = "INSERT INTO avto.log(add_date, n_reg_new) VALUES ('" \
                           + str(today) + "','" + str(value) + "')"
            # print (insert_query)
            cursor.execute(insert_query)
            connection.commit()

            select_query = "SELECT * FROM avto.avto WHERE n_reg_new = UPPER('" \
                           + value + "') ORDER by d_reg::timestamp DESC LIMIT 1"
            # SQL_Query = "SELECT * FROM avto.avto_2015 WHERE n_reg_new = 'АІ8487СО'"
            # bot.reply_to(message, value)
            cursor.execute(select_query)
            mobile_records = cursor.fetchall()
            # print(SQL_Query)
            # print(mobile_records)
            for row in mobile_records:
                textQuery = "Дата реєстрації: " + row[4] + ",\r\n" \
                            "Бренд: " + row[7] + ",\r\n" \
                            "Модель: " + row[8] + ",\r\n" \
                            "Рік випуску: " + row[9] + ",\r\n" \
                            "Колір: " + row[10] + ",\r\n" \
                            "Паливо: " + row[14] + ",\r\n" \
                            "Об'єм двигуна: " + row[15] + " см3,\r\n" + \
                            "Назва операції: " + row[3]
                # print(textQuery)
                # bot.reply_to(message, textQuery)

            if caption is not None:
                if caption[0:1] == '@':
                    insert_query = "INSERT INTO avto.user(number, chat_id) VALUES ('" \
                                   + str(value) + "','" + str(chat_id) + "')"
                    cursor.execute(insert_query)
                    connection.commit()
                    bot.reply_to(message, "Зареєстровано з номером " + value)
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

        cursor.close()
        connection.close()

        # bot.reply_to(message, "Якщо номер розпізнано не вірно введіть його текстом з використанням кирилиці")

    except Exception as e:
        bot.reply_to(message, str(e))
        print(e)


@bot.message_handler(content_types=['video'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        mylist = []

        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'D:/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        # video_path = 'D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/videos/20190830_092645.mp4'

        #capture = cv2.VideoCapture(src)
        capture = cv2.VideoCapture(src)

        size = (
             int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        # print (capture)

        # codec = cv2.VideoWriter_fourcc(*'DIVX')
        # output = cv2.VideoWriter('D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/videos/out/videofile_masked.avi',
        #                          codec, 60.0, size)

        frame_count = 0
        # batch_size = 1
        scale = 1

        while (capture.isOpened()):
            ret, frame = capture.read()
            frame_count += 1
            if ret:
                # print (frame)

                if frame_count == scale:
                    # name = '{0}.jpg'.format(frame_count - batch_size)
                    # name = os.path.join('D:/TEMP/_deeplearning/__from_kiev/_photo_from_bot/videos/out/', name)

                    # frame = cv2.flip(frame, 0)

                    (h, w, d) = frame.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, 270, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h))

                    # cv2.imwrite(name, frame)
                    # output.write(frame)

                    img = frame
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

                    mylist.append(textArr)

                    scale = scale + 7

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # mylist = sorted(set(map(tuple, mylist)), reverse=True)
        mylist = sort_and_deduplicate(mylist)
        print (mylist)
        for value in mylist:
            bot.reply_to(message, value)

    except Exception as e:
        bot.reply_to(message, e)


@bot.message_handler(content_types=['audio'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'D:/_deeplearning/__from_kiev/_photo_from_bot/' + file_info.file_path;
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

        src = 'D:/_deeplearning/__from_kiev/_photo_from_bot/documents/' + message.document.file_name;
        print(src)
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Document adding")

    except Exception as e:
        bot.reply_to(message, e)


# handler func=lambda
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    chat_id = message.chat.id
    print (chat_id)
    # bot.reply_to(message, message.text)
    connection = psycopg2.connect(user="postgres",
                                  password="postgres",
                                  host="192.168.33.89",
                                  port="5432",
                                  database="geo_new_3")
    cursor = connection.cursor()

    d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
         "T": "Т", "X": "Х"}

    value = replace_all(message.text.upper(), d)
    today = datetime.now()

    if value[0:1] == '@':
        insert_query = "INSERT INTO avto.user(number, chat_id) VALUES ('" \
                       + str(value[1:]) + "','" + str(chat_id) + "')"
        cursor.execute(insert_query)
        connection.commit()
        bot.reply_to(message, "Зареєстровано з номером" + value[1:])

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

        # exit()

#    insert_query = "INSERT INTO avto.log(add_date, n_reg_new) VALUES ('" \
#                   + str(today) + "','" + str(value) + "')"
    # print (insert_query)
#    cursor.execute(insert_query)
#    connection.commit()

#    select_query = "SELECT * FROM avto.avto WHERE n_reg_new = UPPER('" \
#                   + value + "') ORDER by d_reg::timestamp DESC LIMIT 1"
    # SQL_Query = "SELECT * FROM avto.avto_2015 WHERE n_reg_new = 'АІ8487СО'"
#    bot.reply_to(message, value)
#    cursor.execute(select_query)
#    mobile_records = cursor.fetchall()
    # print(SQL_Query)
    # print(mobile_records)
#    for row in mobile_records:
#        textQuery = "Дата реєстрації: " + row[4] + ",\r\n" \
#                    "Бренд: " + row[7] + ",\r\n" \
#                    "Модель: " + row[8] + ",\r\n" \
#                    "Рік випуску: " + row[9] + ",\r\n" \
#                    "Колір: " + row[10] + ",\r\n" \
#                    "Паливо: " + row[14] + ",\r\n" \
#                    "Об'єм двигуна: " + row[15] + " см3,\r\n" + \
#                    "Назва операції: " + row[3]
#        print(textQuery)
#        bot.reply_to(message, textQuery)


#    cursor.close()
#    connection.close()

    # bot.reply_to(message, "Розробляється")


bot.polling(none_stop=True, interval=0, timeout=5)