import telebot
import time

bot = telebot.TeleBot("1383386139:AAGWeMlF9BW26ZwUwnVuk2pQm6nOvUADxyw")


@bot.message_handler(commands=['help', 'start'])
def handle_command_adminwindow(message):
    bot.send_message(chat_id=message.chat.id,
                     text="Choose language")


while True:
    try:
        bot.polling(none_stop=True, interval=0, timeout=0)
    except:
        time.sleep(10)
