from tensorflow.python.keras.backend import set_session
import numpy as np
import tensorflow as tf
from config import Config as cfg
import time
import os
from bert_scripts import run_reg
from log_server import log_server

BERT_MODEL = cfg.bert_model
tokenization = run_reg.tokenization
MAX_SEQ_LENGTH = 128
is_lower_case = True
vocab_file = 'bert_scripts/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=is_lower_case)

class Magpie(object):
    def __init__(self, model_file, sess):
        self.sess = sess
        if not model_file:
            self.bert_model = None
        else:
            log_server.logging('>>>>>>>> Load Model {}'.format(model_file))
            self.bert_model = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_file)    

    def predict_score(self, data):
        #0 get tensor
        input_ids = self.sess.graph.get_tensor_by_name('input_ids:0')
        input_mask = self.sess.graph.get_tensor_by_name('input_mask:0')
        segment_ids = self.sess.graph.get_tensor_by_name('segment_ids:0')

        y = self.sess.graph.get_tensor_by_name('loss/Squeeze:0')
        
        #1 create example
        input_example = run_reg.InputExample(guid="", text_a = data, text_b = None, label="0")

        #2 convert to features
        input_features = run_reg.convert_single_example(1, input_example, MAX_SEQ_LENGTH, tokenizer)

        #3 predict
        y_pred = self.sess.run(y,
                        feed_dict={
                            input_ids: np.reshape(input_features.input_ids, (1, MAX_SEQ_LENGTH)),
                            input_mask: np.reshape(input_features.input_mask, (1, MAX_SEQ_LENGTH)),
                            segment_ids: np.reshape(input_features.segment_ids, (1, MAX_SEQ_LENGTH))
                        })   
        return float(y_pred[0])

class Predictor(object):
    def __init__(self, grade):
        self.grade = grade
        tf_config = tf.ConfigProto()
        memory_list = list(map(int, os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | awk '{print $3}'").readlines()))
        memory_total = memory_list[0]
        memory_limited = 5000.0        # use 5G memory
        tf_config.gpu_options.per_process_gpu_memory_fraction = memory_limited / memory_total 
        
        self.sess = tf.Session(config=tf_config)
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = self.create_model()

    def create_model(self):
        magpie = Magpie(model_file=BERT_MODEL[self.grade], sess=self.sess)
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
        super(JuniorOnePredictor, self).__init__('??????')

class JuniorTwoPredictor(Predictor):
    def __init__(self):
        super(JuniorTwoPredictor, self).__init__('??????')

class JuniorThreePredictor(Predictor):
    def __init__(self):
        super(JuniorThreePredictor, self).__init__('??????')

class SeniorOneoPredictor(Predictor):
    def __init__(self):
        super(SeniorOneoPredictor, self).__init__('??????')

class SeniorTwoPredictor(Predictor):
    def __init__(self):
        super(SeniorTwoPredictor, self).__init__('??????')

class SeniorThreePredictor(Predictor):
    def __init__(self):
        super(SeniorThreePredictor, self).__init__('??????')

class AESInference(object):
    """
    The AES system
    """
    def __init__(self):
        self.grade_list = ["??????", "??????","??????","??????","??????", "??????","??????"]
        self.model_map = {
            '??????': JuniorOnePredictor,
            '??????': JuniorTwoPredictor,
            '??????': JuniorThreePredictor,
            '??????': JuniorThreePredictor,
            '??????': SeniorOneoPredictor,
            '??????': SeniorTwoPredictor,
            '??????': SeniorThreePredictor
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
        ????????????????????????
        :param data: ??????
        :param grade: ????????????
        :return: ??????????????????
        """
        if data is None:
            return 0
        
        result = self.models[grade].predict(data)
        return result
