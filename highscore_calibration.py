#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author : dengyu
# @Time   : 2019/09/25

from sklearn.externals import joblib
import numpy as np
import math

linear_models = [
    './model/linear/junior_1_model_degree_2.model',
    './model/linear/junior_2_model_degree_2.model',
    './model/linear/junior_3_model_degree_2.model'
]
poly_linear_models = [
    './model/linear/junior_1_poly_linear_model_degree_2.model',
    './model/linear/junior_2_poly_linear_model_degree_2.model',
    './model/linear/junior_3_poly_linear_model_degree_2.model',
    './model/linear/senior_1_poly_linear_model_degree_1.model',
    './model/linear/senior_2_poly_linear_model_degree_1.model',
    './model/linear/senior_3_poly_linear_model_degree_1.model'
]

def fit(machine_score, grade):
    """
    分数矫正
    :param machine_score: 机评分数 0 - 1 之间
    :param grade: 年级：0 - 5 分别对应初一到高三
    :return: 矫正后的分数
    """
    def gen_feats(machine_score):
        assert 0 <= machine_score <= 1
        return [math.exp(machine_score - 0.7), machine_score]

    assert 0 <= grade <= 5
    feats = gen_feats(machine_score)
    feats = np.array([[item] for item in feats]).reshape(1, -1)
    degree = 2 if grade <= 2 else 1
    if degree == 1:
        poly_linear_model = joblib.load(poly_linear_models[grade])
        y_predict = poly_linear_model.predict(feats)
    elif degree == 2:
        model = joblib.load(linear_models[grade])
        poly_linear_model = joblib.load(poly_linear_models[grade])
        x_quadratic = model.fit_transform(feats)   # TODO order报错
        y_predict = poly_linear_model.predict(x_quadratic)
    return y_predict[0]

if __name__ == '__main__':
    print(fit(0.76, 2))
