# coding=utf-8
import nltk
from collections import defaultdict
from treelib import Node, Tree
from stanfordcorenlp import StanfordCoreNLP
#from config import SentencePattern, ConstituencyParse, Be
import logging
import re
#import preprocess
import argparse
import os
from enum import Enum

class SentencePattern (Enum):

      clause             = 1     # 从句句型
      therebe            = 2     # There be句型
      comparative        = 3     # 比较句型
      passive            = 4     # 被动句型
      present_participle = 5     # 现在分词独立结构句型
      subjunctive        = 6     # 虚拟语气句型
      multi              = 7     # 混合句型
      unknown            = 0     # 未知句型

ConstituencyParse = [
      "ROOT", "IP", "NP", "VP", "PU", "LCP", "PP", "CP",
      "DNP", "ADVP", "ADJP", "DP", "QP", "NN", "NR", "NT",
      "PN", "VV", "VC", "CC", "VE", "VA", "AS", "VRD", "T",
      "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
      "LS", "MD", "PDT", "POS", "PRP", "RB", "RBR", "RBS",
      "RP", "SYM", "TO", "WDT", "WP", "WP&", "WRB", "NNS",
      "NNP", "NNPS", "PDT", "PP$", "S", "SBAR", "SBARQ",
      "SINV", "SQ", "WHADVP", "WHNP", "WHPP", "X", "*", "0",
      "VBG", "VBN", "VB", "PRP$", "VBP", "VBZ", "VBD",
      "WHADJP", "UH", "INTJ", "FRAG",
]

Be = ["am", "is", "are", "was", "were", "been", "be", "being",
      "Am", "Is", "Are", "Was", "Were", "Been", "Be", "Being",
      "'m",]

class PatternClassifier():
    def __init__(self, stanfordnlp_path=r'./stanford-corenlp-full-2016-10-31'):
        self.stanfordCoreNLP = StanfordCoreNLP(stanfordnlp_path)

    def gen_pattern_feats(self, essay):
        """
        生成句型分类特征
        :param essay: 作文内容
        :param nlp: 句法解析器
        :return: 返回该作文的句型列表 如：[1, 0, 0, 0, 0, 0, 3, 3]
        """
        pattern_list = []
        essay_list = nltk.sent_tokenize(essay)
        result_final_dict = reset_result_dict()
        for i in range(len(essay_list)):
            sentence = essay_list[i]
            try:
                parse_str = self.stanfordCoreNLP.parse(sentence)
            except Exception as e:
                #print(str(e))
                continue
            sentence_infer_results = infer_pattern(sentence, parse_str)
            if sentence_infer_results is not None:
                for item in sentence_infer_results:
                    pattern = item["pattern"]
                    if pattern == SentencePattern.clause:
                        result_final_dict["clause"] = result_final_dict["clause"] + 1
                    if pattern == SentencePattern.therebe:
                        result_final_dict["therebe"] = result_final_dict["therebe"] + 1
                    if pattern == SentencePattern.comparative:
                        result_final_dict["comparative"] = result_final_dict["comparative"] + 1
                    if pattern == SentencePattern.passive:
                        result_final_dict["passive"] = result_final_dict["passive"] + 1
                    if pattern == SentencePattern.present_participle:
                        result_final_dict["present_participle"] = result_final_dict["present_participle"] + 1
                    if pattern == SentencePattern.subjunctive:
                        result_final_dict["subjunctive"] = result_final_dict["subjunctive"] + 1
                    if pattern == SentencePattern.multi:
                        result_final_dict["multi"] = result_final_dict["multi"] + 1
                    if pattern == SentencePattern.unknown:
                        result_final_dict["unknown"] = result_final_dict["unknown"] + 1
        pattern_list.append(result_final_dict['clause'])
        pattern_list.append(result_final_dict['therebe'])
        pattern_list.append(result_final_dict['comparative'])
        pattern_list.append(result_final_dict['passive'])
        pattern_list.append(result_final_dict['present_participle'])
        pattern_list.append(result_final_dict['subjunctive'])
        pattern_list.append(result_final_dict['multi'])
        pattern_list.append(result_final_dict['unknown'])
        return pattern_list

    def close_parser(self):
        self.stanfordCoreNLP.close()


# 找到be动词在解析句中的索引
def findBeIndex(parse_list):
    be_indexs = []
    for index, item in enumerate(parse_list):
        if item in Be:
            be_indexs.append({str(index): item})
    return be_indexs

# 找相同的元素
def findDuplicates(parse_list, tag):
    source = parse_list
    tally = defaultdict(list)
    for i, item in enumerate(source):
        tally[item].append(i)
    for item in tally:
        if item == tag:
            return tally[item]
    return []

# 构造句法结构树
def build_parser_tree(org_str, parse_str):
    level_dict = defaultdict(list)
    punctuation = "',.!?[]()%#@&1234567890"
    parse_tree = Tree()
    parse_list = parse_str.replace("(", " ( ").replace(")", " ) ").strip().split()
    #print(parse_list)
    #print(org_str)
    org_str_list = nltk.word_tokenize(org_str)
    left_bracket_counter = 0
    right_bracket_counter = 0
    level = 0
    for index, item in enumerate(parse_list):
        if item == "(":
            left_bracket_counter = left_bracket_counter + 1
            continue
        if item == ")":
            right_bracket_counter = right_bracket_counter + 1
            continue
        level = left_bracket_counter - right_bracket_counter
        try:
            if (item in ConstituencyParse and item not in punctuation) \
                    or (item not in org_str and item.isupper()):
                # 创建非叶子节点，如：ROOT、S、VP、NP
                if item == "ROOT":
                    parse_tree.create_node(item, str(index))
                    level_dict[str(level)].append(str(index))
                else:
                    parse_tree.create_node(item, str(index), parent=level_dict[str(level-1)][-1])
                    level_dict[str(level)].append(str(index))
            if item in org_str_list and item not in punctuation:
                # 创建叶子节点，每一个叶子节点都是句子中的一个单词
                parse_tree.create_node(item, str(index), parent=level_dict[str(level)][-1])
                level_dict[str(level+1)].append(str(index))
            elif item in punctuation:
                # 创建标点符号的相关节点
                if index > 0 and parse_list[index-1] == "(":
                    parse_tree.create_node(item, str(index), parent=level_dict[str(level-1)][-1])
                    level_dict[str(level)].append(str(index))
                else:
                    parse_tree.create_node(item, str(index), parent=level_dict[str(level)][-1])
                    level_dict[str(level+1)].append(str(index))
        except Exception as e:
            #print(str(e))
            return None, parse_list
    #parse_tree.show()

    # 叶子节点中有词性标签，说明建树出错
    for node in parse_tree.leaves():
        if node.tag in ConstituencyParse:
            print("建立句法树出错！")
            return None, parse_list
    return parse_tree, parse_list

# 判断从句句型
"""
规则：词性为"SBAR"
"""
def infer_clause(org_str, parse_list, parse_tree):
    dict = {}
    list = []
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown
    if "SBAR" in parse_list:
        leaves = parse_tree.leaves(str(parse_list.index("SBAR")))
        if len(leaves) > 0:
            leaves.sort(key=lambda x: int(x.identifier), reverse=False)
            #print("该句是从句:", " ".join([item.tag for item in leaves]))
            dict["pattern"] = SentencePattern.clause
            dict["extra"] = " ".join([item.tag for item in leaves])
            list.append(dict)
    return list

"""
规则：词性中有“EX”，或者有单词“there”、“There”，并且存在be动词，且它们的层级不超过3
"""
# 判断There be句型
def infer_therebe(org_str, parse_list, parse_tree):
    dict = {}
    list = []
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown
    node_id = ""
    if "EX" in parse_list:
        node_id = str(parse_list.index("EX"))
    if "there" in parse_list:
        node_id = str(parse_list.index("there"))
    if "There" in parse_list:
        node_id = str(parse_list.index("There"))
    if node_id == "":
        return list
    # 找到句子中的be动词
    be_index_list = findBeIndex(parse_list)
    if len(be_index_list) > 0:
        # 找到be动词的位置，转成string
        if len(be_index_list) == 1:
            be_node_id = ""
            for dict_key in be_index_list[0].keys():
                be_node_id = dict_key
            if abs(parse_tree.level(be_node_id) - parse_tree.level(node_id)) <= 3:
                dict["pattern"] = SentencePattern.therebe
                #print("There be句型")
                list.append(dict)
        else:
            for be_index_dict in be_index_list:
                be_node_id = ""
                for dict_key in be_index_dict.keys():
                    be_node_id = dict_key
                if abs(parse_tree.level(be_node_id) - parse_tree.level(node_id)) <= 3:
                    dict["pattern"] = SentencePattern.therebe
                    #print("There be句型")
                    list.append(dict)
                    break
    return list

"""
规则：1、存在“JJR”或者“RBR”，且存在“than”（My brother is three years older than I．）
      2、“so...as”或者“as ... as”（He is as good a cook as Peter.）
      3、“more than”或者“less than”（No less than 100 people offered to buy it.）
      4、as + adj + a/an + noun + as 结构（He is as good a cook as Peter.）
      5、“the same ... as ...”
      6、存在“JJS”、“RBS”
"""
# 判断比较句型
def infer_comparative(org_str, parse_list, parse_tree):
    dict = {}
    list = []
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown
    if "JJR" in parse_list or "RBR" in parse_list or "than" in parse_list: #规则1
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    if re.search(r'(as|so) (.*?) as', org_str, re.M | re.I): #规则2
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    if re.search(r'(more|less|More|Less) than', org_str, re.M | re.I): #规则3
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    if re.search(r'as (.*?) (a|an) (.*?) as', org_str, re.M | re.I):#规则4
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    if re.search(r'the same (.*?) as', org_str, re.M | re.I):#规则5
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    if "JJS" in parse_list or "RBS" in parse_list:#规则6
        dict["pattern"] = SentencePattern.comparative
        #print("比较句型")
        list.append(dict)
        return list
    return list


"""
规则：be动词 + 动词过去式(VBN)，且be动词在过去式动词的前面
"""
# 判断被动句型
def infer_passive(org_str, parse_list, parse_tree):
    dict = {}
    list_result = []
    org_str_list = nltk.word_tokenize(org_str)
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown
    if "VBN" in parse_list:
        # 找到过去式动词的索引，有可能有多个?
        vbn_indexes_parse_list = findDuplicates(parse_list, "VBN")
        if len(vbn_indexes_parse_list) > 0:
            # 在原句中的索引位置
            try:
                vbn_indexes_org_list = [org_str_list.index(parse_list[int(index) + 1]) for index in vbn_indexes_parse_list]
            except Exception as e:
                print("VBN not found.")
                return []
            # 找到be动词的索引
            be_index_list = findBeIndex(org_str_list)
            if len(be_index_list) > 0:
                if len(be_index_list) == 1:
                    be_index = 0
                    for dict_key in be_index_list[0].keys():
                        be_index = int(dict_key)

                    for vbn_index in vbn_indexes_org_list:
                        if be_index < vbn_index:
                            dict["pattern"] = SentencePattern.passive
                            list_result.append(dict)
                            #print("被动句型")
                            return list_result
                else:
                    for be_index_dict in be_index_list:
                        be_index = 0
                        for dict_key in be_index_dict.keys():
                            be_index = int(dict_key)

                        for vbn_index in vbn_indexes_org_list:
                            if be_index < vbn_index:
                                dict["pattern"] = SentencePattern.passive
                                list_result.append(dict)
                                #print("被动句型")
                                return list_result
    return list_result


# 判断现在分词独立结构句型
"""
规律：独立结构中的名词或代词和其后边的现在分词在逻辑上为主谓关系，即名词或代词为现在分词动作的执行者。
规则：1、存在“,”，且存在“VBG”，并且在“,”分割开的所在段落中唯一的动词
      2、存在词性“VBG”，且“VBG”的祖父节点是“VP”，“VP”的兄弟节点是“NP”
      (Weather permitting，we'll go out for a picnic tomorrow.
        We explored the caves, Peter acting as guide. )
"""
def infer_present_participle(org_str, parse_list, parse_tree):
    dict = {}
    list = []
    org_str_list = nltk.word_tokenize(org_str)
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown

    if "VBG" in parse_list:
        # 规则1
        if "," in parse_list:
            # 找到“VBG”所在的那一部分
            vbg_node_index = parse_list.index("VBG")
            comma_index = parse_list.index(",")
            # 在后面的段落
            if vbg_node_index > comma_index:
                for index, parse_tag in enumerate(parse_list[comma_index:]):
                    if "VB" in parse_tag and parse_tag != "VBG":
                        return list
                    elif index == len(parse_list[comma_index:])-1:
                        dict["pattern"] = SentencePattern.present_participle
                        #print("现在分词独立句型")
                        list.append(dict)
                        return list

            else: # 前面段落
                for index, parse_tag in enumerate(parse_list[:comma_index]):
                    if "VB" in parse_tag and parse_tag != "VBG":
                        return list
                    elif index == len(parse_list[:comma_index])-1:
                        dict["pattern"] = SentencePattern.present_participle
                        #print("现在分词独立句型")
                        list.append(dict)
                        return list
        # 规则2
        # 找到“VBG”的索引
        vbg_node_index = parse_list.index("VBG")
        if vbg_node_index < len(parse_list) - 1:
            # 找到“VBG”的叶子节点，即对应的单词，如 permitting、acting
            vbg_str = parse_list[vbg_node_index + 1]
            vbg_str_index_in_org_str = org_str_list.index(vbg_str)
            if vbg_str_index_in_org_str != 0:
                # 找到原句子中过去分词单词前一个单词，一般为名词，如 Weather、Peter
                np_str = org_str_list[vbg_str_index_in_org_str - 1]
                # 得到该名词的祖父节点
                try:
                    parent = parse_tree.parent(str(parse_list.index(np_str)))
                    grandparent = parse_tree.parent(parent.identifier)
                except Exception:
                    return list
                # “VBG”的父节点是否是“VP”且上面得到的名词的祖父节点是否是“NP”
                if parse_tree.parent(str(vbg_node_index)).tag == "VP" and grandparent.tag == "NP":
                    dict["pattern"] = SentencePattern.present_participle
                    #print("现在分词独立句型")
                    list.append(dict)
                    return list
    return list


"""
规律：
     如果存在If/if，就表示该句是虚拟语气。
     条件句谓语: 过去完成时 + 主句的谓语: would/might/could/should + have + 过去分词。 
      （If you had arrived a little earlier, you would have seen her.）
     条件句谓语: 动词用过去式 + 主句谓语: would/might/could + 动词原形。
     （If you left your bag outside, someone would steal it. ）
     条件句中常用的一种句型是 have not been for, 意思为 “要不是”。
     （If it had not been for your help, I would have failed. ）
     条件句也可以采用倒装语序。
     （Had we got there earlier, we would have caught the train.）
     条件句采用 were to + 不定式
    （If it were to rain today, we would not go for a picnic.）
    条件句采用 should + 不定式
    （If you should have any difficulty in writing this report, you could come to my office.）
     条件句采用If it were not for 句式
    （If it were not for their help, we would be in serious trouble. ）
    条件句使用倒装语序
    （Were you in my position, you would do the same.）
    would rather/ 'd rather
    If only
规则：
"""
# 判断虚拟语气句型
def infer_subjunctive(org_str, parse_list, parse_tree):
    dict = {}
    list = []
    dict["text"] = org_str.strip()
    dict["pattern"] = SentencePattern.unknown
    org_str_list = nltk.word_tokenize(org_str)
    if re.search(r'(If|if)', org_str, re.M | re.I):
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if "SINV" in parse_list:
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'(would|wouldn\'t|mightn\'t|might|could|couldn\'t|should|shouldn\'t) have', org_str, re.M | re.I) or \
       re.search(r'(would|might|could|should) not have', org_str, re.M | re.I):
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if "VBD" in parse_list and "MD" in parse_list:
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'(had|have|has) not been for', org_str, re.M | re.I) and "MD" in parse_list:
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'were to', org_str, re.M | re.I) and "MD" in parse_list:
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'(If|if) it were not for', org_str, re.M | re.I) and "MD" in parse_list:
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'would rather', org_str, re.M | re.I) or ("'d" in parse_list and "rather" in parse_list):
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    if re.search(r'(If|if) only', org_str, re.M | re.I):
        dict["pattern"] = SentencePattern.subjunctive
        list.append(dict)
        #print("虚拟语气")
        return list
    return list

# 推断句型
def infer_pattern(org_str, parse_str):
    parse_tree, parse_list = build_parser_tree(org_str, parse_str)
    # 如果解析树出错，则返回未知类型
    if parse_tree is None:
        result_tmp = []
        dict = {"pattern": SentencePattern.unknown, "text": org_str}
        result_tmp.append(dict)
        #print("未知句型")
        return result_tmp
    result = infer_clause(org_str, parse_list, parse_tree)
    result.extend(infer_therebe(org_str, parse_list, parse_tree))
    result.extend(infer_comparative(org_str, parse_list, parse_tree))
    result.extend(infer_passive(org_str, parse_list, parse_tree))
    result.extend(infer_present_participle(org_str, parse_list, parse_tree))
    result.extend(infer_subjunctive(org_str, parse_list, parse_tree))
    result_tmp = []
    for item in result: # item是字典，{"pattern":xx, "text":xx, "extra":xx}
        if len(item) > 0:
            result_tmp.append(item)
    # 未知句型
    if len(result_tmp) == 0:
        dict = {"pattern": SentencePattern.unknown, "text": org_str}
        result_tmp.append(dict)
        #print("未知句型")
    elif len(result_tmp) > 1:
        result_tmp = []
        dict = {"pattern": SentencePattern.multi, "text": org_str}
        result_tmp.append(dict)
        #print("混合句型")

    #print(result_tmp)
    return result_tmp

def reset_result_dict():
    result_final_dict = {}
    result_final_dict.setdefault("clause", 0)
    result_final_dict.setdefault("therebe", 0)
    result_final_dict.setdefault("comparative", 0)
    result_final_dict.setdefault("passive", 0)
    result_final_dict.setdefault("present_participle", 0)
    result_final_dict.setdefault("subjunctive", 0)
    result_final_dict.setdefault("multi", 0)
    result_final_dict.setdefault("unknown", 0)
    return result_final_dict

def cli_main(args):
    paths = [os.path.join(args.dir, path) for path in os.listdir(args.dir)]
    import multi_threads
    multi_threads.run_multi_threads(paths)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Sentence pattern classifier')
    # parser.add_argument('--dir', type=str, default='')
    # args = parser.parse_args()
    # cli_main(args)
    essay = "I bought English dictionary from your internet bookstore last weekend. But when I accepted the dictionary, I found that there were some questions in it. First, this dictionary has some misprint, then, it also lasts some pages bring from po is to park and so on. There is a famous saying : ' ' Honesty is the best policy : So I apply for draw backing or change a new one. I would appreciate very much if you can take my application into account. And it is very kind of you to consider my suggestions. Thank you very much !"
    nlp_path = r'./stanford-corenlp-full-2016-10-31'
    classifier = PatternClassifier(stanfordnlp_path=nlp_path)
    pattern_list = classifier.gen_pattern_feats(essay)
    print(pattern_list)
