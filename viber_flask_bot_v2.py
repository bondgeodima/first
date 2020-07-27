import flask
from flask import Flask, request, Response
from viberbot import Api
from viberbot.api.bot_configuration import BotConfiguration
from viberbot.api.messages import VideoMessage
from viberbot.api.messages.text_message import TextMessage
import logging
import json

from viberbot.api.viber_requests import ViberConversationStartedRequest
from viberbot.api.viber_requests import ViberFailedRequest
from viberbot.api.viber_requests import ViberMessageRequest
from viberbot.api.viber_requests import ViberSubscribedRequest
from viberbot.api.viber_requests import ViberUnsubscribedRequest

from flask import send_file

from viberbot.api.messages import (
    TextMessage,
    ContactMessage,
    PictureMessage,
    VideoMessage
)
from viberbot.api.messages.data_types.contact import Contact

import psycopg2

import urllib.request
import time


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

    select_query = "DELETE FROM avto.user WHERE chat_id = " + str(chat_id)

    cursor.execute(select_query)
    connection.commit()


app = flask.Flask(__name__)


viber = Api(BotConfiguration(
    name='NumberCarBot',
    avatar='',
    auth_token='4bdc538a20e7d1b3-21905a4be00e2fd3-e79c68a1197c6ec8'
))


@app.route('/file-downloads/')
def file_downloads():
    try:
        return send_file('D:/_deeplearning/__from_kiev/_photo_from_bot/photos/file_0.jpg',
                         attachment_filename='file_0.jpg')
    except Exception as e:
        return str(e)


@app.route('/', methods=['POST'])
def incoming():
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
            d = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
                 "T": "Т", "X": "Х"}
            print ('send text')
            connection = connection_db()
            cursor = connection.cursor()
            value = replace_all(message.text.upper(), d)
            chat_id = viber_request.sender.id

            if value[0:1] == '@':
                check_register_value = check_register(value, chat_id)

                if check_register_value[0] == "0":
                    insert_query = "INSERT INTO avto.user(number, chat_id, messenger) VALUES ('" \
                                   + str(value[1:]) + "','" + str(chat_id) + "','" + str('v') + "')"
                    cursor.execute(insert_query)
                    connection.commit()
                    # bot.reply_to(message, "Зареєстровано з номером " + value[1:])
                    viber.send_messages(viber_request.sender.id, [
                        TextMessage(text="Зареєстровано з номером " + value[1:])
                    ])
                else:
                    print (check_register_value[1])
                    print (check_register_value[2])
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
                try:
                    select_query = "SELECT number FROM avto.user WHERE chat_id = '" + str(chat_id) + "' LIMIT 1"
                    print(select_query)
                    cursor.execute(select_query)
                    mobile_records = cursor.fetchall()
                    # print(SQL_Query)
                    # print(mobile_records)
                    for row in mobile_records:
                        my_avto_number = row[0]
                        print("my_avto_number = " + my_avto_number)

                    v = message.text.split(' ')
                    avto_number_out = v[0][1:]
                    mes = ' '.join(v[1:])
                    print("avto_number_out " + avto_number_out)
                    print("message" + mes)

                    select_query = "SELECT chat_id FROM avto.user WHERE number = UPPER('" \
                                   + avto_number_out + "') LIMIT 1"
                    #  + message.text.split('/')[0][1:] + "') LIMIT 1"

                    print(select_query)
                    cursor.execute(select_query)
                    mobile_records = cursor.fetchall()
                    # print(SQL_Query)
                    # print(mobile_records)
                    for row in mobile_records:
                        chat_id_out = row[0]
                        print(chat_id_out)
                        # bot.reply_to(message, textQuery)
                    # bot.send_message(chat_id=chat_id, text='#' + avto_number + ' # ' + message.text.split('/')[1])
                    # bot.send_message(chat_id=chat_id_out, text='#' + my_avto_number + ' ' + mes)
                    viber.send_messages(chat_id_out, [
                        TextMessage(text='#' + my_avto_number + ' ' + mes)
                    ])
                except Exception as e:
                    viber.send_messages(viber_request.sender.id, [
                        TextMessage(text=str(e))
                    ])

            if value[0:1] == '&':
                delete_registration(chat_id)
                viber.send_messages(viber_request.sender.id, [
                    TextMessage(text='Реєстрація видалена')
                ])
            else:
                viber.send_messages(viber_request.sender.id, [
                    TextMessage(text=message.text)
                ])
                # exit()

        if t == 'picture':
            print ('send picture')
            print (message)
            caption = message.media.caption
            file_name = str(viber_request.sender.id) + str(time.time()) + str(".jpg")
            urllib.request.urlretrieve(message.media,
                                       "D:/_deeplearning/__from_kiev/_photo_from_bot/photos/viber/"+file_name)
            src = "D:/_deeplearning/__from_kiev/_photo_from_bot/photos/viber/" + file_name
            return Response(status=200)
        # message = PictureMessage(
        #     media="https://0eae6dde0922.ngrok.io/file-downloads/",
        #     text="Viber logo")
        # lets echo back

        # viber.send_messages(viber_request.sender.id, [
        #     message
        # ])
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
    app.run(host='127.0.0.1', port=5000, debug=True, )
    # app.run()

