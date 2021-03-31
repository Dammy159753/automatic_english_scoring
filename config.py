#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author : dengyu
# @Time   : 2019/07/10

"""
The config file of AES system
"""


class Config:
    """
    The config of the AES system
    """
    n_feature = 100
    n_h1 = 50
    n_h2 = 25
    n_h3 = 1
    model_path = './'
    voc_path = './'
    dataroot = './'
    maxlen = 20000
    vocab = None
    tokenize_text = None
    to_lower = True
    num_workers = 16
    batch_size = 64
    shuffle = True
    grade_level_list = [u'四年级', u'五年级', u'六年级', u'初一',
                        u'初二', u'初三', u'初四', u'高一', u'高二', u'高三']  # 年级表
    extra_grade_list = [u'七年级', u'八年级', u'九年级']  # 额外的年级表
    grade_level = 0
    prompt_based = False  # 是否采用作文主题特征 default False
    min_score = 0
    max_score = 100
    nltk_data_flag = False  # 第一次需下载nltk package包, 设为True, 之后改为False
    total_score_junior=15.0
    total_score_senior=25.0
    bert_model = {
        '初一': 'model/bert/junior_1/',   # 初一预测模型地址
        '初二': 'model/bert/junior_2/',   # 初二预测模型地址
        '初三': 'model/bert/junior_3/',   # 初三预测模型地址
        '高一': 'model/bert/senior_1/',   # 高一预测模型地址
        '高二': 'model/bert/senior_2/',   # 高二预测模型地址
        '高三': 'model/bert/senior_3/'    # 高三预测模型地址
    }
    params_by_grade = {
        '初一': {
            'model_path': 'model/lstm/junior_1_len150_vocab20000_lstm_31.h5',   # 初一预测模型地址
            'tokenizer_path': 'model/tokenizer/junior_1_vocab20000.pickle',     # 初一词表
            'essay_len': 150                                                    # 初一作文长度
        },
        '初二': {
            'model_path': 'model/lstm/junior_2_len150_vocab20000_lstm_41.h5',   # 初二预测模型地址
            'tokenizer_path': 'model/tokenizer/junior_2_vocab20000.pickle',     # 初二词表
            'essay_len': 150                                                    # 初二作文长度
        },
        '初三': {
            'model_path': 'model/lstm/junior_3_len150_vocab20000_lstm_51.h5',   # 初三预测模型地址
            'tokenizer_path': 'model/tokenizer/junior_3_vocab20000.pickle',     # 初三词表
            'essay_len': 150                                                    # 初三作文长度
        },
        '高一': {
            'model_path': 'model/lstm/senior_1_len200_vocab20000_lstm_23.h5',   # 高一预测模型地址
            'tokenizer_path': 'model/tokenizer/senior_1_vocab20000.pickle',     # 高一词表
            'essay_len': 200                                                    # 高一作文长度
        },
        '高二': {
            'model_path': 'model/lstm/senior_2_len100_vocab20000_lstm_20.h5',   # 高二预测模型地址
            'tokenizer_path': 'model/tokenizer/senior_2_vocab20000.pickle',     # 高二词表
            'essay_len': 100                                                    # 高二作文长度
        },
        '高三': {
            'model_path': 'model/lstm/senior_3_len150_vocab20000_lstm_14.h5',   # 高三预测模型地址
            'tokenizer_path': 'model/tokenizer/senior_3_vocab20000.pickle',     # 高三词表
            'essay_len': 150                                                    # 高三作文长度
        },
    }
    generate_bin = True  # 是否生成数据的bin文件(加速lightgbm训练)
    ratio = 0.1  # 校正比例, 即采用该比例的试卷数量进行校正
    calib_num = 40  # 校正数量下限
    calib_flag = True  # 校正标志
    feed_back_flag = True  # 是否返回gec的feed back结果 default True
    evaluate_flag = True  # 评价预测性能
    visual_flag = True  # 是否可视化结果
    visual_path = 'visual_1222'  # 可视化结果保存路径
    visual_min_score = 0  # 结果可视化最低分
    visual_max_score = 15  # 结果可视化最高分
    nltk_path = ("./nltk_data")
