# encoding: utf-8
import argparse
import os
import json
import multiprocessing
from multiprocessing import Manager
import time
import math
import pandas as pd
import requests

aes_url = "http://192.168.0.104:9105/AES_post"
def get_aes(post_text, grade):
    post_data = {
        'post_content': post_text, 
        'grade': grade
    }
    while True:
        try:
            r = requests.post(aes_url, data=post_data, verify=False, timeout=2000)
            break
        except Exception as e:
            print(e)
            pass
    res = json.loads(r.text)
    return res

class MultiEvaluator(multiprocessing.Process):
    def __init__(self, essay_list, thread_num, results, thread_id):
        multiprocessing.Process.__init__(self)
        self.essay_list = essay_list
        self.thread_num = thread_num
        self.thread_id = thread_id
        print('thread_id is: ', thread_id)
        self.results = results

    def get_res(self, essay_list):
        for essay in essay_list:
            try:
                if essay[2] == 'junior_1':
                    grade = '初一'
                elif essay[2] == 'junior_2':
                    grade = '初二'
                elif essay[2] == 'junior_3':
                    grade = '初三'
                elif essay[2] == 'senior_1':
                    grade = '高一'
                elif essay[2] == 'senior_2':
                    grade = '高二'
                elif essay[2] == 'senior_3':
                    grade = '高三'
                res = get_aes(essay[6], grade)
                self.results[essay[6]] = res
            except IOError:
                continue

    def run(self):
        thread_group_len = int(math.ceil(len(self.essay_list) / self.thread_num))
        start_point = self.thread_id * thread_group_len
        end_point = start_point + thread_group_len if start_point + thread_group_len < len(
            self.essay_list) else -1
        self.get_res(self.essay_list[start_point:end_point])
        print('start', start_point, end_point)
        print("thread %d finished" % self.thread_id)


def main(arg):
    essay_list = pd.read_csv(arg.test_data_dir).sample(200).values.tolist()
    thread_num = 8
    time_start = time.time()
    with Manager() as manager:
        results = manager.dict()

        work_threads = []
        for i in range(thread_num):
            process = MultiEvaluator(essay_list, thread_num, results, i)
            process.start()
            work_threads.append(process)

        for thread in work_threads:
            thread.join()

        # process results here

    time_end = time.time()
    last_time = time_end - time_start
    fps = len(essay_list) / last_time
    print("process %d essay, cost %d seconds, fps=%d" %
          (len(essay_list), last_time, fps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get aes results")
    parser.add_argument('--test_data_dir',
                        metavar='DIR',
                        default='test/test_data_v3.csv',
                        help='test data set path')
    main(parser.parse_args())