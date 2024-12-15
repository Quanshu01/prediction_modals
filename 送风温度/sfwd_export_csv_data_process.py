import sys
import time
import pandas as pd
import numpy as np
import os
import random

'''
    按照机房划分 
        服务器名列表FWQ
        服务器对应空调KT_toFwq_list
'''
# AB-0,CD-1,EF-2,GH-3,JK-4,LM-5,NP-6
FWQ_202 = ['AB', 'CD', 'EF', 'GH', 'JK', 'LM', 'NP']  # 202机房服务器名
FWQ_203 = ['AB', 'CD', 'EF', 'GH', 'JK']  # 203机房服务器名
FWQ_205 = ['AB', 'CD', 'EF', 'GH', 'JK', 'LM']  # 205机房服务器名
# KT_toFwq_list[0]-->AB列服务器 ... KT_toFwq_list[6]-->NP列服务器
KT_toFwq_list_202 = [['12', '19', '20'], ['10', '11', '18'], ['8', '9', '17'], ['6', '7', '16'],
                     ['4', '5', '15'], ['3', '14'], ['1', '2', '13']]
KT_toFwq_list_203 = [['7', '13'], ['6', '11', '12'], ['4', '5', '10'], ['3', '9'],
                     ['1', '2', '8']]  # 203不向前补0,因为JSON数据导出没有补
KT_toFwq_list_205 = [['1', '10'], ['2', '3', '11'], ['4', '5', '12'], ['6', '13', '14'], ['7', '15'], ['8', '9', '16']]


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def time_func(x):
    '''
        构造时间

        :param x:
        :return: 构造的时间
    '''
    if (x.count(":") == 1):
        return time.mktime(time.strptime(x, "%Y/%m/%d %H:%M"))
    else:
        return time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))

def strtime(x):
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x))

def clean_data(data):
    '''
        清理数据并新增间隔时间列

        :param data: 数据dataframe格式
        :return: 清理后 并 新增新列（与前一行数据时间间隔）
    '''
    # for i in range(20):
    #     data=data[data['KT-'+str(i+1)+'-压缩机控制方式']==3]
    # print(data.shape)
    # diff_time为时间间隔
    data['sampleTimeNum'] = pd.DataFrame(data['sampleTime']).applymap(lambda x: time_func(x))
    data['diff_time'] = data['sampleTimeNum'].diff().values
    data['diff_time'] = data['diff_time'].shift(-1)
    data.replace(np.NaN, 0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def load_csvData(csv_file_name):
    '''
        以绝对路径 加载数据，并按照clean_data清理数据、新增间隔时间列

        :param csv_file_name: csv文件路径
        :return: 经过数据清洗的csv文件路径下的Dataframe数据
    '''
    DATA_DIR = "data"
    data_df = pd.read_csv(os.path.join(DATA_DIR, csv_file_name))
    if ('Unnamed: 0' in list(data_df)):
        data_df = data_df.drop('Unnamed: 0', axis=1)
    return data_df


def get_variables_df(origin_dataset, var_name: str):
    '''
        从获取的csv数据集中 获取 列名为"var_name"关键词的变量 数据表

        :param origin_dataset: 原DataFrame数据
        :param var_name: 需要提取的变量名
        :return: 需要提取的var_name变量列
    '''
    all_variables = list(origin_dataset.columns)  # 列表：输入数据集所有列名
    data_clos = [i for i in all_variables if var_name in i]
    data = origin_dataset[data_clos]
    return data  # 返回含有"var_name"关键词的变量 数据表

def choose_from_20KT(KT_list: list, num: int) -> list:
    '''
        从 KT_list空调组 中每组随机选择 num 台

        :param KT_list: 空调组二维列表
        :param num: 每组随机选择 num 台
        :return: 在KT_list二维空调列表中 随机选择num台空调 的结果
    '''
    choice = []
    for i in range(len(KT_list)):
        choice.append(random.sample(KT_list[i], num))  # 随机从一组服务器中选2个
    return choice  # 返回随机选择的空调列表


def chosen_namelist(list_chosen: list, name: str) -> list:
    '''
        输入：列表和名字

        :param list_chosen: 所选列表
        :param name: 名字
        :return: 含有空调所在服务器、以及空调名的变量名列表
    '''
    name_list = []
    for i in list_chosen:  # 所选机房所有组服务器对应空调 二维列表，两层for取值
        for j in i:
            name_list.append('KT-' + str(j) + "-" + name)
    return name_list  # 返回 含有空调所在服务器、以及空调名的变量名列表


def fwq_close_to(i: int, KT_list: list):
    '''
        输入当前组服务器标号，输出临近2-3组服务器

        :param i: 当前组服务器标号
        :param KT_list: 服务器列表
        :return: 包括自己在内，最近3组服务器（靠墙取临近一列）
    '''
    if i == 0:
        return [i, i + 1]
    elif i == len(KT_list) - 1:
        return [len(KT_list) - 2, len(KT_list) - 1]
    elif i > 0 and i < len(KT_list) - 1:
        return [i - 1, i, i + 1]
    else:
        return "Unvalid FWQ label."  # 其他异常服务器标号

def time_shift(data, n: int):
    '''
        构造 多变量的 时序数据

        :param data: DataFrame数据
        :param n: n分钟时序数据
        :return: 经构造的 时序数据
    '''
    column = data.columns
    for i in range(1, n):
        for j in column:
            data[j + "(-" + str(i) + ")"] = data[j].shift(i)
    data.drop(data.head(n - 1).index, inplace=True, axis=0)
    return data  # 返回此刻及过去n时序数据