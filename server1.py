import json

from flask import Flask, request, make_response, render_template
from flask_cors import CORS

from chatbot import Chatbot

app = Flask(__name__)
CORS(app)  # 启用CORS扩展

# 在全局上下文中挂载一个对象
app.config['bot'] = Chatbot()


@app.route("/api/chat", methods=["POST"])
def chat():
    chatbot = app.config.get("bot")

    user_input = request.json['message']
    msg = chatbot.chat_response(user_input)
    msg = {"text": msg.replace("EOS", "")}

    resp = make_response(json.dumps({"info": msg, "code": 0}, ensure_ascii=False))
    resp.headers["Content-Type"] = "application/json;charset=UTF-8"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/", methods=["get"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=8888)
