import pickle
import re

import jieba
import numpy as np
import requests
from gensim.models import Word2Vec
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed, Activation
from keras.layers import concatenate, dot
from keras.models import Model, load_model
from keras_preprocessing import sequence


class MoodDetect:
    def __init__(self):
        # 加载预训练的 LSTM 模型
        self.model = load_model('model/lstm_java_total.h5')

    def emotion_detection_function(self, input):
        voc_dim = 150
        ######
        # 加载预训练的词向量模型 model_word
        model_word = Word2Vec.load('model/Word2Vec_java.pkl')
        # 始化权重矩阵 embedding_weights
        input_dim = len(model_word.wv.key_to_index.keys()) + 1
        embedding_weights = np.zeros((input_dim, voc_dim))
        # 创建词汇表 w2dic
        w2dic = {}
        for i in range(len(model_word.wv.key_to_index.keys())):
            embedding_weights[i + 1, :] = model_word.wv[list(model_word.wv.key_to_index.keys())[i]]
            w2dic[list(model_word.wv.key_to_index.keys())[i]] = i + 1

        pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
        # 定义情感标签的映射关系
        label = {0: "生气", 1: "伤感", 2: "焦虑", 3: "抑郁"}
        # in_stc=["明天","就要","考试","我","特别","紧张","一点","都","没有","复习"]
        # in_str="我要抑郁死了"
        # 使用正则表达式 pchinese 提取中文字符，然后进行分词。
        # in_stc=''.join(pchinese.findall(in_str))
        in_stc = ''.join(pchinese.findall(input))
        in_stc = list(jieba.cut(in_stc, cut_all=True, HMM=False))

        new_txt = []

        data = []
        for word in in_stc:
            try:
                new_txt.append(w2dic[word])
            except:
                new_txt.append(0)
        data.append(new_txt)

        data = sequence.pad_sequences(data, maxlen=voc_dim)
        pre = self.model.predict(data, verbose=0)[0].tolist()
        return label[pre.index(max(pre))]


class Chatbot:
    def __init__(self):
        K.clear_session()
        self.vocab_size = None
        self.maxLen = None
        self.word_to_index = None
        self.index_to_word = None
        self.question_model = None
        self.answer_model = None

        self.load_data()
        self.build_models()
        self.moodDetect = MoodDetect()

    def load_data(self):
        # question = np.load('pad_question.npy')
        # answer = np.load('pad_answer.npy')
        # answer_o = np.load('answer_o.npy', allow_pickle=True)
        with open('vocab_bag.pkl', 'rb') as f:
            self.words = pickle.load(f)
        with open('pad_word_to_index.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)
        with open('pad_index_to_word.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)

        self.vocab_size = len(self.word_to_index) + 1
        self.maxLen = 20

    def build_models(self):
        truncatednormal = TruncatedNormal(mean=0.0, stddev=0.05)
        embed_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=100,
            mask_zero=True,
            input_length=None,
            embeddings_initializer=truncatednormal
        )
        LSTM_encoder = LSTM(
            512,
            return_sequences=True,
            return_state=True,
            kernel_initializer='lecun_uniform',
            name='encoder_lstm'
        )
        LSTM_decoder = LSTM(
            512,
            return_sequences=True,
            return_state=True,
            kernel_initializer='lecun_uniform',
            name='decoder_lstm'
        )

        input_question = Input(shape=(None,), dtype='int32', name='input_question')
        input_answer = Input(shape=(None,), dtype='int32', name='input_answer')

        decoder_dense1 = TimeDistributed(Dense(256, activation="tanh"))
        decoder_dense2 = TimeDistributed(Dense(self.vocab_size, activation="softmax"))

        input_question_embed = embed_layer(input_question)
        input_answer_embed = embed_layer(input_answer)

        encoder_lstm, question_h, question_c = LSTM_encoder(input_question_embed)

        decoder_lstm, _, _ = LSTM_decoder(input_answer_embed, initial_state=[question_h, question_c])

        attention = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_lstm], axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_lstm])
        output = decoder_dense1(decoder_combined_context)
        output = decoder_dense2(output)

        model = Model([input_question, input_answer], output)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model.load_weights('models/W--184-0.5949-.h5')
        # model.summary()
        self.model = model

        question_model = Model(input_question, [encoder_lstm, question_h, question_c])
        self.question_model = question_model

        answer_h = Input(shape=(512,))
        answer_c = Input(shape=(512,))
        encoder_lstm = Input(shape=(self.maxLen, 512))
        target, h, c = LSTM_decoder(input_answer_embed, initial_state=[answer_h, answer_c])
        attention = dot([target, encoder_lstm], axes=[2, 2])
        attention_ = Activation('softmax')(attention)
        context = dot([attention_, encoder_lstm], axes=[2, 1])
        decoder_combined_context = concatenate([context, target])
        output = decoder_dense1(decoder_combined_context)
        output = decoder_dense2(output)
        answer_model = Model([input_answer, answer_h, answer_c, encoder_lstm], [output, h, c, attention_])
        self.answer_model = answer_model

    def act_weather(self, city):
        url = 'http://wthrcdn.etouch.cn/weather_mini?city=' + city
        page = requests.get(url)
        data = page.json()
        temperature = data['data']['wendu']
        notice = data['data']['ganmao']
        outstrs = "地点： %s\n气温： %s\n注意： %s" % (city, temperature, notice)
        return outstrs + ' EOS'

    def input_question(self, seq):
        seq = jieba.lcut(seq.strip(), cut_all=False)
        sentence = seq
        try:
            seq = np.array([self.word_to_index[w] for w in seq])
        except KeyError:
            seq = np.array([36874, 165, 14625])
        seq = sequence.pad_sequences([seq], maxlen=self.maxLen, padding='post', truncating='post')
        return seq, sentence

    def decode_greedy(self, seq, sentence):
        question = seq
        for index in question[0]:
            if int(index) == 5900:
                for index_ in question[0]:
                    if index_ in [7851, 11842, 2406, 3485, 823, 12773, 8078]:
                        return self.act_weather(self.index_to_word[index_])
        answer = np.zeros((1, 1))
        attention_plot = np.zeros((20, 20))
        answer[0, 0] = self.word_to_index['BOS']
        i = 1
        answer_ = []
        flag = 0
        encoder_lstm_, question_h, question_c = self.question_model.predict(x=question, verbose=0)
        while flag != 1:
            prediction, prediction_h, prediction_c, attention = self.answer_model.predict([
                answer, question_h, question_c, encoder_lstm_
            ], verbose=0)
            attention_weights = attention.reshape(-1, )
            attention_plot[i] = attention_weights
            word_arg = np.argmax(prediction[0, -1, :])  #
            answer_.append(self.index_to_word[word_arg])
            if word_arg == self.word_to_index['EOS'] or i > 40:
                flag = 1
            answer = np.zeros((1, 1))
            answer[0, 0] = word_arg
            question_h = prediction_h
            question_c = prediction_c
            i += 1
        result = ''.join(answer_)
        return result

    def chat_response(self, input):
        """
        仅聊天
        :param input:
        :return:
        """
        # with self.graph.as_default():
        seq, sentence = self.input_question(input)
        answer = self.decode_greedy(seq, sentence)
        return answer

    def chat_moodDetect(self, user_input):
        """
        聊天 + 心情检测
        :param user_input:
        :return:
        """
        chatbot_response = self.chat_response(user_input)
        last_sentence = self.get_last_sentence(user_input)
        emotion = self.moodDetect.emotion_detection_function(last_sentence)
        return {'response': chatbot_response, 'emotion': emotion}

    def get_last_sentence(self, user_input):
        sentences = re.split(r'[.!?]', user_input)
        last_sentence = sentences[-1].strip() if sentences else ''
        return last_sentence
