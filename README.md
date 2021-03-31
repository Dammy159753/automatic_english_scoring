# Introduction

AES(automated essay scoring system) 是现在人工打分模型中最常用，也是最主流的模型。 其中根据文章中的内容结合提取的特征进行打分。
内置了很多功能，包括寻找高亮词（高亮词参照人工提取的高亮词）；可以用高分作文作为标准作为离题检测，同时可以最大程度的忽略prompt特征，做到模型的通用性。本项目更多的介绍在实际应用中对中国学生的英文写作水平的真实打分，分数较客观，要比老师的评分略低一些。

# Requirements and Installation

scikit-learn ==  0.20.1
nltk  
lightgbm  
numpy   
pandas   
minepy  
language-check   
genism
scipy
stanfordcorenlp==3.9.1.1

scikit-learn == 0.20.1

treelib==1.5.5

enchant

Java jdk==1.8


# Requirements Language Package

1) nltk_data 链接[nltk](https://blog.csdn.net/u010167269/article/details/63684137)
包括:corpora, sentiment, taggers, tokenizers
2) 下载stanford_core  
3) 下载word2vec (放在/model下)
4）下载language_check, 连接[LanguageCheck](https://www.languagetool.org/), 解压language_check放在项目里，不要用setup安装，只解压就好。

# Methods
本项目BERT训练AES的方式 ，并改造bert的分类为回归方式


# Data  （TODO）
工程根目录: 

./data/
├── all_vocabulary.json
├── bin_data
├── high_score_new.txt
├── highScore.txt
├── highword_data
│   ├── junior_high_vocab.txt
│   ├── junior_samewords.txt
│   ├── senior_high_vocab.txt
│   └── senior_samewords.txt
├── retrain_data
│   └── add_data
│       ├── complate_senior_No1.json
│       └── eval_senior_No1_new.json
├── samewords.txt
└── test_confusion.json

  
./data 包括了模型训练和预测的全部数据

数据预处理 :
python data_pre_post_process.py    # 将图片地址, ocr识别结果保存为json   
python gen_json_multi_process.py   # 多线程提取文章的features, 然后保存为json格式, 加快训练速度  


训练数据(已经经过ocr的后处理算法):    
在data/retrain/.. 下存放训练数据



# Train & Test

train:         
python train_senior.py 训练高中模型，更改里面的训练数据路径
python train_junior.py 训练初中模型，更改里面的训练数据路径
 
 
test:  
python main.py  



# Api   

接口定义  

采用了flask的Restful API的方式，之后会采用gunicorn的后端启动方法
在AES_server.py


# Notes & Future Work
a. 当前并没有考虑题目特征, 对离题作文的打分不准   
b. 对低年级(如初中)的作文评分效果较差, 总体来说, 打分有些许偏低的趋势  
c. 当前的数据集质量不太高, 会出现一些打分异常的数据, 与老师批改的态度有较大的关系, 批改分数是比较主观的, 并不像图片一样直观    
d. 由于现在对比各种机器学习的方法，采用GBM，之后会曾是使用DL的方法    


# Reference  
[1] Automated essay evaluation with semantic analysis  
[2] Task-Independent Features for Automated Essay Grading  
[3] Automated essay scoring with string kernels and word embeddings
[4] Automated Assessment of Non-Native Learner Essays: Investigating the Role of Linguistic Features  
[5] Coherence-based Automatic Essay Assessment  
[6] Attention-based Recurrent Convolutional Neural Network for Automatic Essay Scoring
[7] 英语作文自动评分算法的研究与设计_刘浩坤
[8] 英语语法自动检错系统的设计与实现
[9] A neural approach to automated essay scoring  https://github.com/nusnlp/nea  
[10] TDNN: A Two-stage Deep Neural Network for Prompt-independent Automated Essay Scoring https://github.com/ucasir/TDNN
[11] A Memory-Augmented Neural Model for Automated Grading  https://github.com/siyuanzhao/automated-essay-grading  
[12] https://github.com/DDigimon/TCAMP-WEEK2  
[13] https://github.com/edx/ease  
