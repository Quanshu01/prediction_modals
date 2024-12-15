import os
import warnings
warnings.filterwarnings("ignore", category=Warning)

from sfwd_export_csv_data_process import load_csvData, get_variables_df, clean_data
from sfwd_export_csv_data_process import choose_from_20KT, chosen_namelist, fwq_close_to, time_shift, strtime
from sfwd_export_csv_data_process import FWQ_202, FWQ_203, FWQ_205, KT_toFwq_list_202, KT_toFwq_list_203, KT_toFwq_list_205
import pandas as pd
import numpy as np
import time

'''
    定义202/203/205机房 所有连续csv文件路径
'''
# 202  csv数据 所在文件夹
csv_file_JF_202 = "E:/qs/"
# 203  csv数据 所在文件夹
csv_file_JF_203 = "E:/qs/"

csv_file_2022 = 'E:/qs/test/2022/'
csv_file_2023 = 'E:/qs/test/2023/'

class params_load():
    def __init__(self, JF_name: str, KT_name: str, past_num: int):
        self.JF_name = JF_name  # 机房名
        self.KT_name = KT_name - 1  # 服务器名
        self.past_num = past_num  #

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

        # 所选机房空调的数目
        self.KT_num = len(list(np.concatenate(self.KT_toFwq_list).reshape((-1, 1), order="F")))
        # 从所选机房所选服务器中, 每组服务器 随机选择 2台
        self.KT_chosen = choose_from_20KT(self.KT_toFwq_list, 2)

    # 导入所有的csv文件
    def all_csvs(self,year=0):
        '''
            导入所有的CSV文件并concat

            :return: concat后的训练数据
        '''
        # 202机房训练csv文件
        if year == 2022:
            csv_file_name = csv_file_2022
        elif year == 2023:
            csv_file_name = csv_file_2023
        elif self.JF_name == '202':
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

    def sfwd_exportfrom_csv(self, data):  # 返回从csv文件导出的 送风温度原始数据 (输入：csv数据+空调数量；输出：20台空调最终送风温度平均数据及对应时间)
        sfwd1_avg = get_variables_df(data, '送风温度1')  # 获得送风温度1的平均
        return sfwd1_avg
    def sfwd_set_exportfrom_csv(self, data):  # 返回从csv文件导出的 送风温度原始数据 (输入：csv数据+空调数量；输出：20台空调最终送风温度平均数据及对应时间)
        sfwd1_avg = get_variables_df(data, '送风温度设定')  # 获得送风温度1的平均
        return sfwd1_avg
    def hfwd_exportfrom_csv(self, data_df):  # 返回从csv文件导出的 送风温度原始数据 (输入：csv数据+空调数量；输出：20台空调最终送风温度平均数据及对应时间)
        hfwd1_avg = get_variables_df(data_df, '回风温度1')  # 获得回风温度1的平均
        hfwd2_avg = get_variables_df(data_df, '回风温度2')  # 获得回风温度2的平均
        hfwd3_avg = get_variables_df(data_df, '回风温度3')  # 获得回风温度3的平均
        hfwd4_avg = get_variables_df(data_df, '回风温度4')  # 获得回风温度4的平均

        col=[co[:-1]+"平均" for co in list(hfwd1_avg)]
        hfwd_avg=pd.DataFrame()
        for i in range(hfwd1_avg.shape[1]):
            hfwd_avg[col[i]]=pd.concat([hfwd1_avg.iloc[:,i], hfwd2_avg.iloc[:,i], hfwd3_avg.iloc[:,i], hfwd4_avg.iloc[:,i]], axis=1).apply(lambda x: x.mean(),
                                                                                  axis=1)
        return hfwd_avg
    def hfwd_set_exportfrom_csv(self, data):  # 返回从csv文件导出的 送风温度原始数据 (输入：csv数据+空调数量；输出：20台空调最终送风温度平均数据及对应时间)
        hfwd_set = get_variables_df(data, '回风温度设定')  # 获得送风温度1的平均
        return hfwd_set

    def zs_or_zsset_exportfrom_csv(self, data, isSET: bool, zs_name: str):
        if isSET == False:
            if zs_name == '风' or zs_name == '冷凝风':
                zs1 = get_variables_df(data, '-' + zs_name + '机1转速')
                zs2 = get_variables_df(data, '-' + zs_name + '机2转速')
            elif zs_name == '压缩':
                zs1 = get_variables_df(data, '-' + zs_name + '机1容量')
                zs2 = get_variables_df(data, '-' + zs_name + '机2容量')
            zs1, zs2 = zs1.iloc[:, :self.KT_num], zs2.iloc[:, :self.KT_num]
            # col = [co[:-3] + "转速平均" for co in list(zs1)]
            # zs_avg = pd.DataFrame()
            # for i in range(zs1.shape[1]):
            #     zs_avg[col[i]] = pd.concat(
            #         [zs1.iloc[:, i], zs2.iloc[:, i]],axis=1).apply(lambda x: x.mean(),axis=1)
            return zs1,zs2  # 返回 该机房 所有空调 转速平均值
        elif isSET == True:
            if zs_name == '风' or zs_name == '冷凝风':
                zs1_set = get_variables_df(data, '-' + zs_name + '机1转速设定')
                zs2_set = get_variables_df(data, '-' + zs_name + '机2转速设定')
            elif zs_name == '压缩':
                zs1_set = get_variables_df(data, '-' + zs_name + '机1容量设定')
                zs2_set = get_variables_df(data, '-' + zs_name + '机2容量设定')
            # col = [co[:-5] + "转速设定平均" for co in list(zs1_set)]
            # zs_set_avg = pd.DataFrame()
            # for i in range(zs1_set.shape[1]):
            #     zs_set_avg[col[i]] = pd.concat(
            #         [zs1_set.iloc[:, i], zs2_set.iloc[:, i]], axis=1).apply(lambda x: x.mean(), axis=1)
            return zs1_set,zs2_set  # 20台空调 风机/压缩机1/2转速设定 数据

    def create_sfwd(self, sfwd1):
        '''
            # 返回sfwd: 第fwq组服务器对应空调及临近服务器对应空调 送风温度1 数据
            :param sfwd1:
            :param isKT:
            :return:
        '''
        sfwd1 = sfwd1.take([self.KT_name], axis=1)
        # if num[0] == 1:
        return sfwd1

    def create_hfwd(self, hfwd_avg):  # 返回hfwd：第KT台空调 回风温度1、4总数据
        hfwd_avg = hfwd_avg.take([self.KT_name], axis=1)
        return hfwd_avg

    def create_sfwd_set(self, sfwd_set):
        sfwd_set = pd.DataFrame(sfwd_set.iloc[:, self.KT_name])  # 得到 第num台空调 送风温度设定
        return sfwd_set
    def create_hfwd_set(self, hfwd_set):
        hfwd_set = pd.DataFrame(hfwd_set.iloc[:, self.KT_name])  # 得到 第num台空调 送风温度设定
        return hfwd_set
    def create_zs_or_zsset(self, zs1_set, zs2_set, isKT: bool):
        '''
              # num参数：第一位表示风机1还是2；第二位表示第KT台空调 或者 第fwq组服务器
            :param zs1_set:
            :param zs2_set:
            :param isKT:
            :return:
        '''
        if isKT == True:  # 第KT台空调特定风机/压缩机转速
            zs1_set = pd.DataFrame(zs1_set.iloc[:, self.KT_name])  # 取 该台空调 风机1/压缩机1 转速设定
            zs2_set = pd.DataFrame(zs2_set.iloc[:, self.KT_name])  # 取 该台空调 风机2/压缩机2 转速设定
            zs_set = pd.concat([zs1_set, zs2_set], axis=1)
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

def all_data(JF_num: str, KT_name: int, past_num: int,swwd_flag,year=0):  # 从csv数据中导出 需要特征 的全部数据
    pl = params_load(JF_name=JF_num, KT_name=KT_name, past_num=past_num)

    # 所在机房 全部数据
    data_df = pl.all_csvs(year=year)  # 导入原始数据
    sfwd1= pl.sfwd_exportfrom_csv(data_df)  # 20台空调 送风温度1/4/总avg 数据
    sfwd_set= pl.sfwd_set_exportfrom_csv(data_df)  # 20台空调 送风温度1/4/总avg 数据
    # hfwd_avg= pl.hfwd_exportfrom_csv(data_df)  # 20台空调 送风温度两个测点 数据
    # hfwd_set= pl.hfwd_set_exportfrom_csv(data_df)  # 20台空调 送风温度两个测点 数据
    # fj1_zs,fj2_zs = pl.zs_or_zsset_exportfrom_csv(data_df, False, '风')  # 所选机房所有组服务器对应空调 风机1转速
    # fj1zs_set,fj2zs_set= pl.zs_or_zsset_exportfrom_csv(data_df, True, '风')  # 所选机房所有组服务器对应空调 风机1转速设定,风机2转速设定 数据
    # ysj1_zs,ysj2_zs= pl.zs_or_zsset_exportfrom_csv(data_df, False, '压缩')  # 20台空调 风机1转速
    # ysj1zs_set,ysj2zs_set= pl.zs_or_zsset_exportfrom_csv(data_df, True, '压缩')  # 20台空调 风机1转速设定,风机2转速设定 数据
    # swwd=get_variables_df(data_df,'室外环境温度')
    print("get all sfwd prediction need csv data")


    # 所选服务器 对应变量
    sfwd = pl.create_sfwd(sfwd1)  # 第fwq组服务器对应空调 及 临近服务器对应空调 送风温度1总数据
    # hfwd = pl.create_hfwd(hfwd_avg)
    # fjzs = pl.create_zs_or_zsset(fj1_zs,fj2_zs, True)
    # ysjzs = pl.create_zs_or_zsset(ysj1_zs,ysj2_zs, True)
    # fjzs_set = pl.create_zs_or_zsset(fj1zs_set,fj2zs_set, True)
    # ysjzs_set = pl.create_zs_or_zsset(ysj1zs_set,ysj2zs_set, True)
    sfwd_set = pl.create_sfwd_set(sfwd_set)
    # hfwd_set = pl.create_hfwd_set(hfwd_set)
    diff_time = get_variables_df(data_df, 'diff_time')
    print("get  %s fwq  prediction all need data" % pl.KT_name)

    # 获取 输入模型变量
    if(swwd_flag==False):
        features = pd.concat([sfwd, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
        # features = pd.concat([sfwd, hfwd, fjzs, ysjzs, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
    else:
        features = pd.concat([sfwd, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息
        # features = pd.concat([sfwd, hfwd, fjzs, ysjzs,swwd, diff_time], axis=1)  # 2022.11.25 删除 送风温度所有平均值，更改为功率信息

    features = time_shift(features, past_num)
    features.reset_index(drop=True, inplace=True)  # ，否则后面concat出现NAN
    print("get all past times-series variabels")

    set = pd.concat([sfwd_set], axis=1)  # 送风温度和回风温度 设定
    # set = pd.concat([sfwd_set,hfwd_set, fjzs_set, ysjzs_set], axis=1)  # 送风温度和回风温度 设定
    set = set.drop(set.head(past_num - 1).index)  # ，确保concat时含有 过去15的时序数据 以及 当前时刻的送风/回风温度设定
    set.reset_index(drop=True, inplace=True)
    print("get all now this time variabels")

    # 2022.12.4 更换为当前时刻变量
    features = pd.concat([features, set], axis=1)  # concat时含有过去15的时序数据以及当前时刻的送风/回风温度设定
    # features = pd.concat([features], axis=1)  # concat时含有过去15的时序数据
    X = pd.DataFrame(features)
    print(features.columns)
    # y = features["KT-" + str(KT_name) + "-送风温度1的avg"]  # labels
    for col in features.columns:
        if f"KT-{KT_name}-" in col and "送风温度" in col:
            y = features[col]
            break
    else:
        raise KeyError(f"在数据中未找到包含'送风温度'且以'KT-{KT_name}-'开头的列")
    # possible_col_names = [
    #     "KT-" + str(KT_name) + "-送风温度1",
    #     "KT-" + str(KT_name) + "-送风温度1的avg"
    # ]
    # for col_name in possible_col_names:
    #     if col_name in features.columns:
    #         y = features[col_name]
    #         break
    # else:
    #     raise KeyError(f"给定的可能列名 {possible_col_names} 均不存在于数据中")
    print("generate features-X and labels-y successfully!")
    print('original features-X and labels-y, X:%s, y:%s' % (str(X.shape), str(y.shape)))
    print("labels:", y)
    return X, y

def deal_odd_data(JF_num: str, KT_name: int, next_num,past_num,train_X_data_csv: str,train_y_data_csv: str,swwd_flag,year=0):
    pl = params_load(JF_name=JF_num, KT_name=KT_name, past_num=past_num)
    X, y = all_data( JF_num, KT_name, past_num,swwd_flag,year=year)
    y = y[next_num:]
    y.reset_index(drop=True, inplace=True)
    y = y.to_frame(name="target")
    deal_odd_data = pd.concat([X, y], axis=1)

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
    if(JF_num=='203'):
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='EF')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='GH')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-3-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-4-')))]
        deal_odd_data = deal_odd_data[deal_odd_data.columns.drop(list(deal_odd_data.filter(regex='KT-5-')))]




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