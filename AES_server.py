#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author : dengyu
# @Time   : 2019/06/10

from flask import Flask, jsonify, request, make_response, abort
from highlight_words import AdvancedWords
import numpy as np
import time
from AES_BERT import AESInference
from log_server import log_server

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = True

# 高分拟合映射关系
grade_dict = {"初一": 0, "初二": 1,"初三": 2,"初四":2,"高一": 3,"高二": 4,"高三": 5}
model = AESInference()

@app.route('/AES_post', methods=['POST'])
def AES_post():
    log_server.logging('============AES Begin!===========')
    content = request.values.get('post_content')
    grade = request.values.get('grade')

    if grade in list(grade_dict.keys()):
        start = time.time()
        AD = AdvancedWords()

        # 模型预测分数方法（predict代表机评分）
        try:
            # TODO: predict = model.infer(......) (DL predict)
            predict = model.infer(content, grade)
        except Exception as e:
            predict = 0
        
        # 高亮词提取和高亮词数
        try:
            highlight_site, high_num = AD.find_highlight_site(content, grade)
        except Exception as e:
            highlight_site = []
        
        predict_ = {'predict': predict, 'highlight': highlight_site}
        log_server.logging('>>>>>>>> Time of Predict: {:.2f}'.format(time.time()-start))
        log_server.logging('>>>>>>>> Predict Score: {}'.format(predict))
        log_server.logging("===============AES Over!!!==============" + "\n")
        return jsonify(predict_)
    else:
        predict, highlight_site = 0, []
        predict_ = {'predict': predict, 'highlight': highlight_site}
        log_server.logging('>>>>>>>> Invalid Grade')
        log_server.logging("===============AES Over!!!==============" + "\n")
        return jsonify(predict_)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=10000)
    app.run()
