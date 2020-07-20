import telebot
import ast
import time
from telebot import types
import psycopg2

bot = telebot.TeleBot("1383386139:AAGWeMlF9BW26ZwUwnVuk2pQm6nOvUADxyw")

stringList = {"ua": "Український", "ru": "Русский", "eng": "Englishe", "fr": "Française", "it": "Italiano",
              "de": "Deutsche", "es": "Español", "pt": "Português", "ar": "عرب", "chi": "中文", "ja": "日本人"}

chat_id = ''

def makeKeyboard():
    markup = types.InlineKeyboardMarkup()

    for key, value in stringList.items():
        markup.add(types.InlineKeyboardButton(text=value,
                                              callback_data="['value', '" + value + "', '" + key + "']"))

    return markup


def returnHelp(leng):

    print (leng)

    connection = psycopg2.connect(user="postgres",
                                  password="postgres",
                                  host="192.168.33.89",
                                  port="5432",
                                  database="geo_new_3")
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
    print (help_text)

    bot.send_message(chat_id=147353364, text=help_text[0:1000])

    # return help_text

# @bot.message_handler(commands=['help', 'start'])
# def send_welcome(message):
 
#     bot.reply_to(message, description)

@bot.message_handler(commands=['test'])
def handle_command_adminwindow(message):
    bot.send_message(chat_id=message.chat.id,
                     text="Choose language",
                     reply_markup=makeKeyboard(),
                     parse_mode='HTML')
    chat_id = message.chat.id
    user_name = message.from_user.username
    user_id = message.from_user.id
    print (chat_id, user_name, user_id)


@bot.callback_query_handler(func=lambda call: True)
def handle_query(call):

    if (call.data.startswith("['value'")):
        print(f"call.data : {call.data} , type : {type(call.data)}")
        print(f"ast.literal_eval(call.data) : {ast.literal_eval(call.data)} , type : "
              f"{type(ast.literal_eval(call.data))}")
        valueFromCallBack = ast.literal_eval(call.data)[1]
        keyFromCallBack = ast.literal_eval(call.data)[2]
        returnHelp(keyFromCallBack)
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


@bot.message_handler(commands=['number'])  # Объявили ветку для работы по команде <strong>number</strong>
def phone(message):
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)  # Подключаем клавиатуру
    button_phone = types.KeyboardButton(text="Отправить телефон",
                                        # Указываем название кнопки, которая появится у пользователя
                                        request_contact=True)
    keyboard.add(button_phone)  # Добавляем эту кнопку
    bot.send_message(message.chat.id, 'Номер телефона',
                     # Дублируем сообщением о том, что пользователь сейчас отправит боту
                     # свой номер телефона (на всякий случай, но это не обязательно)
                     reply_markup=keyboard)


@bot.message_handler(content_types=[
    # Объявили ветку, в которой прописываем логику на тот случай, если пользователь решит прислать номер телефона :)
    'contact'])
def contact(message):
    if message.contact is not None:  # Если присланный объект <strong>contact</strong> не равен нулю
        print(
            # Выводим у себя в панели контактные данные. А вообщем можно их, например, сохранить или сделать что-то еще.
            message.contact)


while True:
    try:
        bot.polling(none_stop=True, interval=0, timeout=0)
    except:
        time.sleep(10)
