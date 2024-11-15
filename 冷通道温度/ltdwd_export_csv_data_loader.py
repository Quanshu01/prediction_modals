import os
import warnings
warnings.filterwarnings("ignore", category=Warning)

from ltdwd_export_csv_data_process import load_csvData, get_variables_df, clean_data
from ltdwd_export_csv_data_process import choose_from_20KT, chosen_namelist, fwq_close_to, time_shift, strtime
from ltdwd_export_csv_data_process import FWQ_202, FWQ_203, FWQ_205, KT_toFwq_list_202, KT_toFwq_list_203, KT_toFwq_list_205
import pandas as pd
import numpy as np
import time

'''
    定义202/203/205机房 所有连续csv文件路径
'''
# 202  csv数据 所在文件夹
csv_file_JF_202="E:/lry/模型汇总/JF_202_test/"
# 203  csv数据 所在文件夹
csv_file_JF_203="E:/lry/模型汇总/JF_203/"

class params_load():
    def __init__(self, JF_name: str, FWQ_name: str, past_num: int, maxORavg: str):
        self.JF_name = JF_name  # 机房名
        self.FWQ_name = FWQ_name  # 服务器名
        self.past_num = past_num
        self.maxORavg = maxORavg  # 预测max还是avg

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
        if self.JF_name=='203':
            self.KT_num=13
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
        # 205 新数据，数据文件和202/203机房不是同时导出，所以文件数量不同，在此return
        # elif self.JF_name == '205':
        #     csv_file_name0 = csv_file_name_JF205_0
        #     csv_file_name1 = csv_file_name_JF205_1
        #     csv_file_name2 = csv_file_name_JF205_2
        #     csv0 = load_csvData(csv_file_name0)
        #     csv1 = load_csvData(csv_file_name1)
        #     csv2 = load_csvData(csv_file_name2)
        #     csv = pd.concat([csv0, csv1, csv2], axis=0)
        #     csv.reset_index(drop=True, inplace=True)
        #     print("JF" + self.JF_name + " all original data shape:", csv.shape)
        #     return csv
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

    def ltdwd_exportfrom_csv(self, data):
        '''
            获取 所选机房所有服务器 冷通道温度均值

            :param data: DataFrame数据
            :return: 所选机房所有服务器 冷通道温度均值
        '''
        ltdwd_original_data = get_variables_df(data,
                                               '冷通道温度的' + self.maxORavg)  # 从 原始csv文件 中提取 所选机房所有服务器 冷通道温度均值数据
        JF_need_FWQ_name = chosen_namelist(self.KT_chosen, '冷通道温度的' + self.maxORavg)
        ltdwd_fliter_JF_need_FWQKT = ltdwd_original_data[JF_need_FWQ_name]  # 从原20台空调中筛选得到 所选机房所有服务器 冷通道温度均值数据
        ltdwd_fliter_fwq = ltdwd_fliter_JF_need_FWQKT.iloc[:, 0]
        for i in range(0, 2 * len(self.KT_toFwq_list), 2):  # 循环实现求得 该机房所有组服务器 的 冷通道温度均值
            ltdwd_i_avg = ltdwd_fliter_JF_need_FWQKT.take([i, i + 1], axis=1).apply(lambda x: x.mean(),
                                                                                    axis=1)  # 求每台空调的 冷通道温度均值
            ltdwd_i_avg = pd.DataFrame(ltdwd_i_avg).rename(
                columns={0: self.FWQ[int(i / 2)] + "组服务器" + "冷通道温度的" + self.maxORavg})  # 修改列名
            ltdwd_fliter_fwq = pd.concat([ltdwd_fliter_fwq, ltdwd_i_avg], axis=1)  # 每台空调的平均送风温度
        ltdwd_fliter_fwq = ltdwd_fliter_fwq.iloc[:, 1:]
        return ltdwd_fliter_fwq

    def sfwd_exportfrom_csv(self, data):  # 返回从csv文件导出的 送风温度原始数据 (输入：csv数据+空调数量；输出：20台空调最终送风温度平均数据及对应时间)
        # if self.JF_name == '202':
        #     avg = '的avg'  # 202机房送风温度 有后缀avg
        # elif self.JF_name == '203' or self.JF_name == '205':
        #     avg = ''  # 203/205机房送风温度 无后缀avg
        sfwd1_avg = get_variables_df(data, '送风温度1')  # 获得送风温度1的平均
        return sfwd1_avg

    def zs_or_zsset_exportfrom_csv(self, data, isSET: bool, zs_name: str):
        if isSET == False:
            if zs_name == '风' or zs_name == '冷凝风':
                zs1 = get_variables_df(data, '-' + zs_name + '机1转速')
                zs2 = get_variables_df(data, '-' + zs_name + '机2转速')
            elif zs_name == '压缩':
                zs1 = get_variables_df(data, '-' + zs_name + '机1容量')
                zs2 = get_variables_df(data, '-' + zs_name + '机2容量')
            zs1, zs2 = zs1.iloc[:, :self.KT_num], zs2.iloc[:, :self.KT_num]
            return zs1, zs2  # 返回 该机房 所有空调 风机1/2 转速
        elif isSET == True:
            if zs_name == '风' or zs_name == '冷凝风':
                zs1_set = get_variables_df(data, '-' + zs_name + '机1转速设定')
                zs2_set = get_variables_df(data, '-' + zs_name + '机2转速设定')
            elif zs_name == '压缩':
                zs1_set = get_variables_df(data, '-' + zs_name + '机1容量设定')
                zs2_set = get_variables_df(data, '-' + zs_name + '机2容量设定')
            return zs1_set, zs2_set  # 20台空调 风机/压缩机1/2转速设定 数据

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

    def create_zs_or_zsset(self, zs1_set, zs2_set, isKT: bool):
        '''
              # num参数：第一位表示风机1还是2；第二位表示第KT台空调 或者 第fwq组服务器
            :param zs1_set:
            :param zs2_set:
            :param isKT:
            :return:
        '''
        num = [0, self.fwq]
        if isKT == True:  # 第KT台空调特定风机/压缩机转速
            zs1_set = pd.DataFrame(zs1_set.iloc[:, num[1]])  # 取 该台空调 风机1/压缩机1 转速设定
            zs2_set = pd.DataFrame(zs2_set.iloc[:, num[1]])  # 取 该台空调 风机2/压缩机2 转速设定
            if num[0] == 1:
                zs_set = zs1_set  # num参数第一位表示风机1,取到风机1转速
            elif num[0] == 2:
                zs_set = zs2_set  # num参数第一位表示风机2,取到风机2转速
        elif isKT == False:  # 第fwq组服务器特定风机转速
            zs1_set_1 = pd.DataFrame()
            zs2_set_2 = pd.DataFrame()

            zs1_set_1 = pd.concat(
                [zs1_set_1, pd.DataFrame(zs1_set.take([int(x) - 1 for x in self.KT_toFwq_list[self.fwq]], axis=1))],
                axis=1)  # 该台空调的 风机1转速设定
            zs2_set_2 = pd.concat(
                [zs2_set_2, pd.DataFrame(zs2_set.take([int(x) - 1 for x in self.KT_toFwq_list[self.fwq]], axis=1))],
                axis=1)  # 该台空调的 风机2转速设定
            zs_set = pd.concat([zs1_set_1, zs2_set_2], axis=1)
        return zs_set

    def create_sfwd_set(self, sfwd_set, isKT: bool):
        '''

            :param sfwd_set:
            :param isKT:
            :return:
        '''
        # 2022.11.6 修改送风温度设定 为该组服务器及临近服务器 对应空调
        if isKT == True:
            sfwd_set_ = pd.DataFrame(sfwd_set.iloc[:, self.fwq])  # 得到 第num台空调 送风温度设定
        elif isKT == False:
            sfwd_set_ = pd.DataFrame(sfwd_set.iloc[:, 0])
            for i in fwq_close_to(self.fwq, self.KT_toFwq_list):  # 得到 该台服务器及临近服务器对应空调送风温度设定
                sfwd_set_i = sfwd_set.take([int(x) - 1 for x in self.KT_toFwq_list[i]],
                                           axis=1)  # 得到 第num组服务器对应空调 送风温度设定
                sfwd_set_ = pd.concat([sfwd_set_, sfwd_set_i], axis=1)
            sfwd_set_ = sfwd_set_.iloc[:, 1:]
        return sfwd_set_

    def create_sfwd(self, sfwd1, isKT: bool):
        '''
            # 返回sfwd: 第fwq组服务器对应空调及临近服务器对应空调 送风温度1 数据
            :param sfwd1:
            :param isKT:
            :return:
        '''
        num = [0, self.fwq]
        if isKT == False:
            sfwd1_ = pd.DataFrame()
            allFWQ = fwq_close_to(num[1], self.KT_toFwq_list)  # 包括自己在内的 所有服务器标号 列表
            for i in allFWQ:  # 得到 该台服务器对应空调送风温度1、4 及 临近服务器对应空调送风温度1、4
                fwq_list = [int(x) - 1 for x in self.KT_toFwq_list[i]]
                sfwd1_i = sfwd1.take(fwq_list, axis=1)  # 取到 第i组服务器对应空调 的 送风温度1数值
                sfwd1_ = pd.concat([sfwd1_, sfwd1_i], axis=1)
            return sfwd1_  # 仅需要1个送风温度
        elif isKT == True:
            sfwd1 = sfwd1.take([num[1]], axis=1)
            if num[0] == 1:
                return sfwd1

    def create_ltdwd(self, ltdwd):
        ltdwd = ltdwd.take([self.fwq], axis=1)  # 第fwq组服务器 冷通道温度数据
        return ltdwd

    def create_powerload(self, power_load):
        power_load = power_load.take([self.fwq], axis=1)
        return power_load


def all_data(maxORavg: str, JF_num: str, FWQ_name: str, past_num: int,swwd_flag):  # 从csv数据中导出 需要特征 的全部数据
    pl = params_load(JF_name=JF_num, FWQ_name=FWQ_name, past_num=past_num, maxORavg=maxORavg)

    # 所在机房 全部数据
    data_df = pl.all_csvs()  # 导入原始数据

    ltdwd = pl.ltdwd_exportfrom_csv(data_df)  # 从原始csv数据 获取 所选机房所有组服务器 冷通道温度最大值均值
    power_load = pl.fwqgl_exportfrom_csv(data_df)  # 从原始csv数据 获取 所选机房所有组服务器功率
    sfwd = pl.sfwd_exportfrom_csv(data_df)  # 所选机房所有组服务器对应空调 送风温度1/4/总avg 数据
    fj1_zs, fj2_zs = pl.zs_or_zsset_exportfrom_csv(data_df, False, '风')  # 所选机房所有组服务器对应空调 风机1转速
    sfwd_set = get_variables_df(data_df, '送风温度设定')  # 获取 该机房 所有空调 送风温度设定
    fj1zs_set, fj2zs_set = pl.zs_or_zsset_exportfrom_csv(data_df, True, '风')  # 所选机房所有组服务器对应空调 风机1转速设定,风机2转速设定 数据
    swwd=get_variables_df(data_df,'室外环境温度')
    print("get all ltdwd prediction need csv data")

    # 所选服务器 对应变量
    ltdwd = pl.create_ltdwd(ltdwd)  # 第fwq组服务器 冷通道温度数据
    power_load = pl.create_powerload(power_load)
    sfwd = pl.create_sfwd(sfwd, False)  # 第fwq组服务器对应空调 及 临近服务器对应空调 送风温度1总数据
    fj_zs = pl.create_zs_or_zsset(fj1_zs, fj2_zs, False)
    sfwd_set = pl.create_sfwd_set(sfwd_set, False)  # 得到 第fwq组服务器对应空调 及 临近服务器对应空调 送风温度设定
    fjzs_set = pl.create_zs_or_zsset(fj1zs_set, fj2zs_set, False)
    diff_time = get_variables_df(data_df, 'diff_time')
    print("get  %s fwq  prediction all need data" % pl.FWQ_name)

    # 获取 输入模型变量
    # features = pd.concat([ltdwd, power_load, sfwd, fj_zs, diff_time, sampleTime], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
    if(swwd_flag==False):
        features = pd.concat([ltdwd, power_load, sfwd, fj_zs, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
    else:
        features = pd.concat([ltdwd, power_load, sfwd, fj_zs,swwd, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
    #
    # features['sampleTime'] = pd.DataFrame(features['sampleTimeNum']).applymap(lambda x: strtime(x))
    # features['diff_time'] = features['sampleTimeNum'].diff().values
    # features['diff_time'] = features['diff_time'].shift(-1)

    features = time_shift(features, past_num)
    features.reset_index(drop=True, inplace=True)


    print("get all past times-series variabels")

    set = pd.concat([sfwd_set, fjzs_set], axis=1)  # 送风温度和回风温度 设定
    set = set.drop(set.head(past_num - 1).index)

    set.reset_index(drop=True, inplace=True)
    print("get all now this time variabels")

    # 2022.12.4 更换为当前时刻变量
    features = pd.concat([features, set], axis=1)  # concat时含有过去15的时序数据以及当前时刻的送风/回风温度设定
    #features = pd.concat([features], axis=1)  # concat时含有过去15的时序数据
    X = pd.DataFrame(features)
    y = features[pl.FWQ_name + "组服务器冷通道温度的" + maxORavg]  # labels # "AB服务器冷通道温度的avg"/max
    print("generate features-X and labels-y successfully!")
    print('original features-X and labels-y, X:%s, y:%s' % (str(X.shape), str(y.shape)))
    print("labels:", y)
    return X, y


def deal_odd_data(maxORavg: str, JF_num: str, FWQ_name:str, next_num,past_num,train_X_data_csv: str,train_y_data_csv: str,swwd_flag):
    pl = params_load(JF_name=JF_num, FWQ_name=FWQ_name, past_num=past_num, maxORavg=maxORavg)
    X, y = all_data(maxORavg, JF_num, FWQ_name, past_num,swwd_flag)
    y = y[next_num:]
    y.reset_index(drop=True, inplace=True)
    y = y.to_frame(name="target")
    deal_odd_data = pd.concat([X, y], axis=1)
    # 如果是203机房，删去所有与空调5有关的变量
    # 处理异常值
    # 判断diff_time，前14个，均小于14才行
    deal_odd_data['delete'] = 0
    # 筛训练数据中不连续的部分（diff_time字段的值）
    for i in range(len(deal_odd_data)):
        for j in range(past_num-1):
            if (deal_odd_data['diff_time(-' + str(j + 1) + ')'][i]) > 120 :
                deal_odd_data['delete'][i] = 1  # 为1则删除
                break
    deal_odd_data = deal_odd_data[deal_odd_data['delete'] != 1]
    deal_odd_data = deal_odd_data[deal_odd_data.columns.drop('delete')]
    deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='diff_time')))]
    #如果是203机房，删除EF/GH列对应的全部变量,对应空调是：3，4，5，9，10
    if(JF_num=='203'):
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='EF')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='GH')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-3-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-4-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-5-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-9-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-10-')))]


    deal_odd_data.reset_index(drop=True, inplace=True)
    deal_odd_data = deal_odd_data.loc[
        (deal_odd_data > 0).all(axis=1) & (deal_odd_data < 1000).all(axis=1)]
    deal_odd_data.reset_index(drop=True, inplace=True)
    deal_odd_data = deal_odd_data.dropna(axis=0)
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
