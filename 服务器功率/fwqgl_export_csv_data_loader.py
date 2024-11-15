import os
import warnings
warnings.filterwarnings("ignore")
from fwqgl_export_csv_data_process import load_csvData, get_variables_df, clean_data
from fwqgl_export_csv_data_process import choose_from_20KT, chosen_namelist, fwq_close_to, time_shift, strtime
from fwqgl_export_csv_data_process import FWQ_202, FWQ_203, FWQ_205, KT_toFwq_list_202, KT_toFwq_list_203, KT_toFwq_list_205
import pandas as pd
import numpy as np
import time

'''
    定义202/203/205机房 所有连续csv文件路径
'''
# 202  csv数据 所在文件夹
csv_file_JF_202="E:/lry/模型汇总/JF_202/"
# 203  csv数据 所在文件夹
csv_file_JF_203="E:/lry/模型汇总/JF_203/"

class params_load():
    def __init__(self, JF_name: str, FWQ_name: str, past_num: int):
        self.JF_name = JF_name  # 机房名
        self.FWQ_name = FWQ_name  # 服务器名
        self.past_num = past_num


        # 机房名不同，服务器名列表FWQ、服务器对应空调KT_toFwq_list 均不同
        if self.JF_name == '202':
            self.FWQ = FWQ_202
            self.KT_toFwq_list = KT_toFwq_list_202
        elif self.JF_name == '203':
            self.FWQ = FWQ_203
            self.KT_toFwq_list = KT_toFwq_list_203
        elif JF_name == '205':
            self.FWQ = FWQ_205
            self.KT_toFwq_list = KT_toFwq_list_205

        # 所选服务器标号
        self.fwq = self.FWQ.index(self.FWQ_name)
        # 所选机房空调的数目
        self.KT_num = len(list(np.concatenate(self.KT_toFwq_list).reshape((-1, 1), order="F")))
        # 从所选机房所选服务器中, 每组服务器 随机选择 2台
        self.KT_chosen = choose_from_20KT(self.KT_toFwq_list, 2)

    # 导入所有的csv文件
    def all_csvs(self):
        '''
            导入所有的CSV文件并concat

            :return: concat后的训练数据
        '''
        # 202机房训练csv文件
        if self.JF_name == '202':
            csv_file_name=csv_file_JF_202
        # 203机房训练csv文件
        elif self.JF_name == '203':
            csv_file_name=csv_file_JF_203
        csv_data=pd.DataFrame()

        data_list = os.listdir(csv_file_name)
        for i in data_list:
            csv_=load_csvData(csv_file_name+i)
            csv_data =pd.concat([csv_data,csv_],axis=0)
        csv_data.reset_index(drop=True, inplace=True)
        csv_data=clean_data(csv_data)
        print("JF" + self.JF_name + " all original data shape:", csv_data.shape)
        return csv_data

    def single_csv(self,csv_file_name):
        csv = load_csvData(csv_file_name)
        return csv

    def fwqgl_exportfrom_csv(self, data):  # 获取 7组服务器 热通道温度均值
        zyggl_original_data = get_variables_df(data, '-总有功功率')  # 从 原始csv文件 中提取 20台空调 总有功功率数据
        yggl_original_data = get_variables_df(data, '-有功功率')  # 从 原始csv文件 中提取 20台空调 有功功率数据

        zyggl_fliter_7KT = zyggl_original_data[
            chosen_namelist(choose_from_20KT(self.KT_toFwq_list, 1), '总有功功率')]  # 从原20台空调中筛选得到7组7台空调 总有功功率数据
        yggl_fliter_7KT = yggl_original_data[
            chosen_namelist(choose_from_20KT(self.KT_toFwq_list, 1), '有功功率')]  # 从原20台空调中筛选得到7组7台空调 有功功率数据

        fwqgl_fliter_7fwq = zyggl_fliter_7KT.iloc[:, 0]
        for i in range(0, len(self.KT_toFwq_list)):  # 循环实现求得 7组服务器 的 功率和
            fwqgl_i = zyggl_fliter_7KT.iloc[:, i] + yggl_fliter_7KT.iloc[:, i]  # 求每台空调的 功率和
            fwqgl_i = pd.DataFrame(fwqgl_i).rename(columns={0: self.FWQ[i] + "组服务器" + "功率"})  # 修改列名
            fwqgl_fliter_7fwq = pd.concat([fwqgl_fliter_7fwq, fwqgl_i], axis=1)  # 每组服务器总功率
        fwqgl_fliter_7fwq = fwqgl_fliter_7fwq.iloc[:, 1:]

        return fwqgl_fliter_7fwq

    def create_powerload(self, power_load):
        power_load = power_load.take([self.fwq], axis=1)
        return power_load


def all_data(JF_num: str, FWQ_name: str, past_num: int,hours_flag):  # 从csv数据中导出 需要特征 的全部数据
    pl = params_load(JF_name=JF_num, FWQ_name=FWQ_name, past_num=past_num)

    # 所在机房 全部数据
    data_df = pl.all_csvs()  # 导入原始数据
    #data_df = pl.single_csv() # 测试单天数据是否可用

    power_load = pl.fwqgl_exportfrom_csv(data_df)  # 从原始csv数据 获取 所选机房所有组服务器功率
    hours=get_variables_df(data_df,'hours')
    print("get all ltdwd prediction need csv data")

    # 所选服务器 对应变量
    power_load = pl.create_powerload(power_load)
    diff_time = get_variables_df(data_df, 'diff_time')
    print("get  %s fwq  prediction all need data" % pl.FWQ_name)

    # 获取 输入模型变量

    features = pd.concat([power_load,diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息

    features = time_shift(features, past_num)
    features.reset_index(drop=True, inplace=True)  # 在past_num的shift后重置索引，否则后面concat出现NAN

    print("get all past times-series variabels")
    if(hours_flag==True):
        features = pd.concat([features,hours], axis=1)  # concat时含有过去15的时序数据
    else:
        features = pd.concat([features], axis=1)  # concat时含有过去15的时序数据
    X = pd.DataFrame(features)
    y = features[pl.FWQ_name + "组服务器功率"]  # labels #
    print("generate features-X and labels-y successfully!")
    print('original features-X and labels-y, X:%s, y:%s' % (str(X.shape), str(y.shape)))
    print("labels:", y)
    return X, y


def deal_odd_data(JF_num: str, FWQ_name:str, next_num,past_num,train_X_data_csv: str,train_y_data_csv: str,hours_flag):
    pl = params_load(JF_name=JF_num, FWQ_name=FWQ_name, past_num=past_num,)
    X, y = all_data( JF_num, FWQ_name, past_num,hours_flag)
    y = y[next_num:]
    y.reset_index(drop=True, inplace=True)
    y = y.to_frame(name="target")
    deal_odd_data = pd.concat([X, y], axis=1)
    fwq_index=pl.FWQ.index(FWQ_name)
    # 如果是203机房，删去所有与空调5有关的变量
    # 处理异常值
    # 判断diff_time，前14个，均小于14才行
    deal_odd_data['delete'] = 0
    # 筛训练数据中不连续的部分（diff_time字段的值）
    for i in range(len(deal_odd_data)):
        for j in range(past_num-1):
            if (deal_odd_data['diff_time(-' + str(j + 1) + ')'][i]) > 120:
                deal_odd_data['delete'][i] = 1  # 为1则删除
                break
    deal_odd_data = deal_odd_data[deal_odd_data['delete'] != 1]
    deal_odd_data = deal_odd_data[deal_odd_data.columns.drop('delete')]
    deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='diff_time')))]
    #202离群服务器值
    if (JF_num == '202'):
        #服务器功率上边界
        gl_upper=[1000,200,200,120,250,200,200]
        # 服务器功率下边界
        gl_lower = [200,160,100,80,180,170,170]


    if(JF_num=='203'):
        gl_upper = [1000,1000,1000,1000,1000 ]
        gl_lower = [0, 0, 0, 0, 0]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='EF')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='GH')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-3-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-4-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-5-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-9-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-10-')))]


    deal_odd_data.reset_index(drop=True, inplace=True)
    if('hours' in deal_odd_data.columns):
        deal_odd_data0 = deal_odd_data[deal_odd_data.columns.drop('hours')]
    else:
        deal_odd_data0 = deal_odd_data
    deal_odd_data = deal_odd_data.loc[
        (deal_odd_data0 > gl_lower[fwq_index]).all(axis=1)&
    (deal_odd_data0 < gl_upper[fwq_index]).all(axis=1)]
    deal_odd_data.reset_index(drop=True, inplace=True)
    deal_odd_data=deal_odd_data.dropna(axis=0)


    print("filter odd features-X and labels-y data successfully!")

    X = deal_odd_data[deal_odd_data.columns.drop('target')]
    y = deal_odd_data['target']
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    print('after dealing with odd data, X:%s, y:%s' % (str(X.shape), str(y.shape)))
    print('variables name:', X.columns)
    X.to_csv(train_X_data_csv,
        index=False,
        header=True)
    # train_olddata_202_LM_ltdwdavg_y
    pd.DataFrame(y).to_csv(
        train_y_data_csv,index=False,
        header=True)
    print("export features-X and labels-y data successfully!")
    return X, y
