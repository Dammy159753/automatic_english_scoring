from AES_DL import AESInference
import numpy as np

request_url = 'http://192.168.0.148:5051/AES_post'

content = "Don't worriy about \nit, I believe you can writing better because of you \nhave a deeply love Chinese's heart. I have some \nadvise to increase Chinese writing ability, hope it \ncan help you. \nFirst, you need to find out material theme : Next, you \nmust writing aroud the material theme. Finally, you can \nwriting about material theme's things. \nI hope this ways can help you step by step \nprogram"
content2 = "More and more people a range lots of \nclasses to improve their children. For example this \npitcure draw a mom is running with her children \nbecause she wants to let her child to study. \nI don't want to see this phenomemoe. Firmly. \nI think children should play in the nature, rather than \nonly study in the class. Secondly, children can not \ntake on these burden, there are too young to study \nlots of knowledge. Last but not least lots of \nclasses will effect children's grow. Ever, they \nwill not enjoy their childhood. \nNow, our country is environment and social are \nthe mainly reasons result this phenoretion, also, because \nparent's expection on their children. \nIn all, parents should not arrange lots of classes \nwe should give children a happy childrard"
content = content2.replace('\n', '')
print(content)
grade = '初三'
machine_score = AESInference.infer(np.array([content]), grade)
print(machine_score)