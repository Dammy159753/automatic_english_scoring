#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author : dengyu
# @Time   : 2019/06/15

import operator
import codecs
import operator
import re
import itertools

class AdvancedWords:

    def __init__(self):
        #self.highlight_word_path = r"./data/high_score_new.txt"
        #self.samewords_path = r"./data/samewords.txt"
        self.senior_highlight_word_path = r"./data/highword_data/senior_high_vocab.txt"
        self.senior_samewords_path = r"./data/highword_data/senior_samewords.txt"
        self.junior_highlight_word_path = r"./data/highword_data/junior_high_vocab.txt"
        self.junior_samewords_path = r"./data/highword_data/junior_samewords.txt"
        self.senior_set = ["高一", "高二", "高三"]
        self.junior_set = ["初一", "初二", "初三", "初四"]

    def load_dictionary(self):
        junior_highlight_list = list()
        junior_synonyms = list()
        senior_highlight_list = list()
        senior_synonyms = list()
        with codecs.open(self.senior_highlight_word_path , mode="r", encoding='utf-8') as f1:
            for line in f1.readlines():
                line = line.strip()
                senior_highlight_list.append(line)
        with codecs.open(self.senior_samewords_path, mode="r", encoding='utf_8') as f2:
            for line in f2:
                senior_synonyms.append(line.strip().split(','))

        with codecs.open(self.junior_highlight_word_path , mode="r", encoding='utf-8') as f1:
            for line in f1.readlines():
                line = line.strip()
                junior_highlight_list.append(line)
        with codecs.open(self.junior_samewords_path, mode="r", encoding='utf_8') as f2:
            for line in f2:
                junior_synonyms.append(line.strip().split(','))
        return senior_highlight_list, senior_synonyms, junior_highlight_list, senior_synonyms

    def remove_synonyms(self, content, grade):
        self.senior_highlight_list, self.senior_synonyms, self.junior_highlight_list, self.junior_synonyms = self.load_dictionary()

        if grade in self.senior_set:
            highscore_words = list()      # 文中所有高亮词的集合
            syn_words = list()            # 在同义词表中的词组集合
            unsyn_words = list()          # 不在同义词表中的词组集合
            remove_syn = list()           # 去掉重复同义词的词组列表
            new_highlight = list()        # 最终生成的新的高亮词表
            for word in self.senior_highlight_list:
                if re.findall(r'(?<= )'+word+r'(?![a-zA-Z])', content):
                    highscore_words.append(word)
            #print("highscore_words:", highscore_words)
            for syn in self.senior_synonyms:
                for phrase in highscore_words:
                    if phrase in syn:
                        syn_words.append(phrase)
                        
                # Handling duplicates in the synonym table
                temp = list()
                for word in syn_words:
                    if word in syn:
                        temp.append(word)
                if operator.ne(len(temp), 0):
                    remove_syn.append(temp[0])
            remove_syn = list(set(remove_syn))
            unsyn_words = [l for l in highscore_words if l not in syn_words]
            senior_new_highlight = unsyn_words + remove_syn
            return senior_new_highlight

        elif grade in self.junior_set:
            highscore_words = list()      # 文中所有高亮词的集合
            syn_words = list()            # 在同义词表中的词组集合
            unsyn_words = list()          # 不在同义词表中的词组集合
            remove_syn = list()           # 去掉重复同义词的词组列表
            new_highlight = list()        # 最终生成的新的高亮词表
            for word in self.junior_highlight_list:
                if re.findall(r'(?<= )' + word + r'(?![a-zA-Z])', content):
                    highscore_words.append(word)
            #print("highscore_words:", highscore_words)
            for syn in self.junior_synonyms:
                for phrase in highscore_words:
                    if phrase in syn:
                        syn_words.append(phrase)
                # Handling duplicates in the synonym table
                temp = list()
                for word in syn_words:
                    if word in syn:
                        temp.append(word)
                if operator.ne(len(temp), 0):
                    remove_syn.append(temp[0])
            remove_syn = list(set(remove_syn))
            unsyn_words = [l for l in highscore_words if l not in syn_words]
            junior_new_highlight = unsyn_words + remove_syn
            return junior_new_highlight
        else:
            return []


    def find_highlight_site(self, content, grade):
        # self.highlight_list, self.synonyms = self.load_dictionary()
        new_highlight = self.remove_synonyms(content, grade)
        repeat_list = list()                                               # Duplicate word list
        for m in range(len(new_highlight)):
            flag = new_highlight[m]
            for n in range(m + 1, len(new_highlight)):
                target = new_highlight[n]
                if flag in target or target in flag:
                    # repeat = "".join([flag[i] for i in range(len(flag)) if flag[i] == target[i]])
                    repeat = self.getNumofCommonSubStr(target, flag)
                    if flag == repeat:
                        repeat_list.append(new_highlight[m])
                        break
                    elif target == repeat:
                        repeat_list.append(new_highlight[n])
                        break
        new_list = list()                                                   # New list of highlighted words
        for every in new_highlight:
            if every not in repeat_list:
                new_list.append(every)

        highlight_site = list()
        high_num = 0
        for high in new_list:
            index = re.search(r'(?<= )'+ high + r'(?!=[a-zA-Z])', content)  # Find the position of the highlighted word in the essay
            position = list(index.span())
            zip_list = dict(zip(high, position))
            highlight_site.append([high, position])
            # highlist_sort = highlight_res.sort(key=lambda elem:elem[1])
            high_num += 1
            if operator.ge(high_num, 10):
                high_num = 10
            elif operator.lt(high_num, 10):
                high_num = high_num
        highlight_site.sort(key=lambda x: int(x[1][0]))
        return highlight_site, high_num

    # Get the largest public string
    def getNumofCommonSubStr(self, str1, str2):
        lstr1 = len(str1)
        lstr2 = len(str2)
        # Occupy space
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        maxNum = 0
        p = 0
        for i in range(lstr1):
            for j in range(lstr2):
                if str1[i] == str2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = i + 1
        return str1[p - maxNum:p]


if __name__ == "__main__":
    #AD = AdvancedWords()
    content = "I be sick in bed take one's as just time eat. However I like take care ice-cream for dinner every much, most but is ready for my mother take care of doesn't want me to eat them, after she is filled with thinks they are as just not healthy a lot. I just do don't like vegetables, especially carrots. And I am made from like milk, too Because I don't want to be fat. I is made from sports very much . I have five rolleyballs and two basketballs . My favorite sport is volley ball . I think it's easy and fun. A fter class. I play rolleyball with my friends. I be good at basket ball, but I don't play it. I only watch it on TV. Soccer is difficult for me , So I don't play it. If you want to benealthy . You can eat healthy food and play sports now."
    grade = '初二'
    #highlight_res, high_num = AD.find_highlight_site(content, grade)
    #print("highlight_site:{} , high_num:{}".format(highlight_res, high_num))

