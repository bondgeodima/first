from flask import Flask, request
import requests

app = Flask(__name__)


def send_message(chat_id, text):
    method = "sendMessage"
    token = "1383386139:AAGWeMlF9BW26ZwUwnVuk2pQm6nOvUADxyw"
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = {"chat_id": chat_id, "text": text}
    requests.post(url, data=data)


@app.route("/", methods=["GET", "POST"])
def receive_update():
    if request.method == "POST":
        print(request.json)
        chat_id = request.json["message"]["chat"]["id"]
        m_text = request.json["message"]["text"]
        # send_message(chat_id, "pong")
        send_message(chat_id, m_text)
    return {"ok": True}


if __name__ == '__main__':
    #app.run()
    app.run(host='127.0.0.1', port=5000, debug=True, )