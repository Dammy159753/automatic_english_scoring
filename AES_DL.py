from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session
from keras.models import load_model
import numpy as np
import pickle
import tensorflow as tf
from config import Config as cfg
import time
import os
from log_server import log_server

PARAMS_BY_GRADE = cfg.params_by_grade

class Magpie(object):
    def __init__(self, model_file, tokenizer_file, max_esssay_len):
        if not model_file:
            self.keras_model = None
        else:
            log_server.logging('>>>>>>>> Load Model {}'.format(model_file))
            self.keras_model = load_model(model_file)

        log_server.logging('>>>>>>>> Load Tokenizer {}'.format(tokenizer_file))
        self.tokenizer = pickle.load(open(tokenizer_file, 'rb'))
        self.max_essay_length = max_esssay_len
    

    def predict_score(self, data):
        sequences = self.tokenizer.texts_to_sequences(data)
        max_essay_length = self.max_essay_length
        data = pad_sequences(sequences, maxlen=max_essay_length, padding='post', truncating='pre')
        y_pred = self.keras_model.predict(data)
        y_pred = np.around(y_pred)[0][0]
        return y_pred

class Predictor(object):
    def __init__(self, grade):
        self.grade = grade
        tf_config = tf.ConfigProto()
        memory_list = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | awk '{print $3}'").readlines()))
        memory_total = memory_list[0]
        # log_server.logging('>>>>>>>> Memory Total: {}'.format(memory_total))
        memory_limited = 5000.0        # use 5G memory
        tf_config.gpu_options.per_process_gpu_memory_fraction = memory_limited / memory_total 
        
        self.sess = tf.Session(config=tf_config)
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = self.create_model()

    def create_model(self):
        magpie = Magpie(model_file=PARAMS_BY_GRADE[self.grade]['model_path'],
                        tokenizer_file=PARAMS_BY_GRADE[self.grade]['tokenizer_path'],
                        max_esssay_len=PARAMS_BY_GRADE[self.grade]['essay_len'])
        return magpie

    def predict(self, data):
        with self.graph.as_default():
            set_session(self.sess)
            if data is None:
                return 0

            predict = self.model.predict_score(data)
        return predict

class JuniorOnePredictor(Predictor):
    def __init__(self):
        super(JuniorOnePredictor, self).__init__('初一')

class JuniorTwoPredictor(Predictor):
    def __init__(self):
        super(JuniorTwoPredictor, self).__init__('初二')

class JuniorThreePredictor(Predictor):
    def __init__(self):
        super(JuniorThreePredictor, self).__init__('初三')

class SeniorOneoPredictor(Predictor):
    def __init__(self):
        super(SeniorOneoPredictor, self).__init__('高一')

class SeniorTwoPredictor(Predictor):
    def __init__(self):
        super(SeniorTwoPredictor, self).__init__('高二')

class SeniorThreePredictor(Predictor):
    def __init__(self):
        super(SeniorThreePredictor, self).__init__('高三')

class AESInference(object):
    """
    The AES system
    """
    def __init__(self):
        self.grade_list = ["初一", "初二","初三","初四","高一", "高二","高三"]
        self.model_map = {
            '初一': JuniorOnePredictor,
            '初二': JuniorTwoPredictor,
            '初三': JuniorThreePredictor,
            '初四': JuniorThreePredictor,
            '高一': SeniorOneoPredictor,
            '高二': SeniorTwoPredictor,
            '高三': SeniorThreePredictor
        }
        self.models = dict()
        load_model_start = time.time()
        for grade in self.grade_list:
            self.models[grade] = self.model_map[grade]()
        
        load_model_end = time.time()
        load_model_total = load_model_end - load_model_start
        log_server.logging('>>>>>>>> Load Model Total Time: {}'.format(load_model_total))

    def infer(self, data, grade):
        """
        机器预测分数接口
        :param data: 作文
        :param grade: 学生年级
        :return: 机器预测分数
        """
        if data is None:
            return 0
        
        result = self.models[grade].predict(data)
        if '初' in grade:
            result = result/cfg.total_score_junior
        elif '高' in grade:
            result = result/cfg.total_score_senior
        else:
            log_server.logging('>>>>>>>> Invalid Grade !!!!!!!!!!!')
        return result
