from flask import Flask, request, Response
from viberbot import Api
from viberbot.api.bot_configuration import BotConfiguration
from viberbot.api.messages import VideoMessage
from viberbot.api.messages.text_message import TextMessage
import logging

from viberbot.api.viber_requests import ViberConversationStartedRequest
from viberbot.api.viber_requests import ViberFailedRequest
from viberbot.api.viber_requests import ViberMessageRequest
from viberbot.api.viber_requests import ViberSubscribedRequest
from viberbot.api.viber_requests import ViberUnsubscribedRequest

from flask import Flask, request
import requests

app = Flask(__name__)

viber = Api(BotConfiguration(
    name='NumberCarBot',
    avatar='',
    auth_token='4bdc538a20e7d1b3-21905a4be00e2fd3-e79c68a1197c6ec8'
))


def send_message(chat_id, text):
    method = "sendMessage"
    token = "1383386139:AAGWeMlF9BW26ZwUwnVuk2pQm6nOvUADxyw"
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = {"chat_id": chat_id, "text": text}
    requests.post(url, data=data)


@app.route('/', methods=['POST'])
def receive_update():
    if request.method == "POST":
        if 'update_id' in request.json:
            # telegram
            print('ok')
            print(request.json)
            chat_id = request.json["message"]["chat"]["id"]
            m_text = request.json["message"]["text"]
            # send_message(chat_id, "pong")
            send_message(chat_id, m_text)
            return {"ok": True}
        else:
            # viber
            print('no')
            logging.debug("received request. post data: {0}".format(request.get_data()))
            # every viber message is signed, you can verify the signature using this method
            if not viber.verify_signature(request.get_data(), request.headers.get('X-Viber-Content-Signature')):
                return Response(status=403)

            # this library supplies a simple way to receive a request object
            viber_request = viber.parse_request(request.get_data())

            if isinstance(viber_request, ViberMessageRequest):
                message = viber_request.message
                # lets echo back
                viber.send_messages(viber_request.sender.id, [
                    message
                ])
            elif isinstance(viber_request, ViberSubscribedRequest):
                viber.send_messages(viber_request.get_user.id, [
                    TextMessage(text="thanks for subscribing!")
                ])
            elif isinstance(viber_request, ViberFailedRequest):
                logging.warn("client failed receiving message. failure: {0}".format(viber_request))

            return Response(status=200)

    # return {"ok": True}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, )
    # app.run()
    # viber.set_webhook('https://0eae6dde0922.ngrok.io')
