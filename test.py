import re

from chatbot import Chatbot
from infer import emotion_detection_function


def chat(user_input):
    chatbot_response = chatbot.chat_response(user_input)
    last_sentence = get_last_sentence(user_input)
    emotion = emotion_detection_function(last_sentence)
    return {'response': chatbot_response, 'emotion': emotion}


def get_last_sentence(user_input):
    sentences = re.split(r'[.!?]', user_input)
    last_sentence = sentences[-1].strip() if sentences else ''
    return last_sentence


if __name__ == '__main__':
    # 创建 Chatbot 实例
    chatbot = Chatbot()
    # moodDetect = MoodDetect()
    while True:
        seq = input("请输入问题：")
        if seq == 'x':
            break
        ans = chatbot.chat_moodDetect(seq)
        print(ans)
