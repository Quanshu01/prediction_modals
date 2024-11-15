import pandas as pd
import os
# 202  csv数据 所在文件夹
csv_file_JF_202="E:/lry/模型汇总/JF_202/"
# 203  csv数据 所在文件夹
csv_file_JF_203="E:/lry/模型汇总/JF_203/"

JF_name='202'
if(JF_name=='202'):
    csv_file_name=csv_file_JF_202
elif(JF_name=='203'):
    csv_file_name = csv_file_JF_203

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

csv_data=pd.DataFrame()
data_list = os.listdir(csv_file_name)
for i in data_list:
    csv_=load_csvData(csv_file_name+i)
    csv_data =pd.concat([csv_data,csv_],axis=0)
csv_data.reset_index(drop=True, inplace=True)
csv_data.to_csv('all_data.csv')
#单独获取送风和冷通道温度，服务器功率


ltdwd = get_variables_df(csv_data,'冷通道温度的avg')  # 从原始csv数据 获取 所选机房所有组服务器 冷通道温度最大值均值
power_load = get_variables_df(csv_data,'服务器功率')  # 从原始csv数据 获取 所选机房所有组服务器功率
sfwd = get_variables_df(csv_data,'送风温度1')  # 所选机房所有组服务器对应空调 送风温度1/4/总avg 数据
sfwd_set = get_variables_df(csv_data, '送风温度设定')  # 获取 该机房 所有空调 送风温度设定

data = pd.concat([ltdwd, power_load, sfwd,sfwd_set], axis=1)
data.to_csv('送风-冷通道.csv')



































#删除列，改名字
# for i in range(1,21):
#     data=data.drop('KT-'+str(i)+'-回风温度1的max', axis=1)
#     data=data.drop('KT-'+str(i)+'-回风温度1的min', axis=1)
#     data = data.drop('KT-' + str(i) + '-回风温度2的max', axis=1)
#     data = data.drop('KT-' + str(i) + '-回风温度2的min', axis=1)
#     data=data.drop('KT-'+str(i)+'-回风温度3的max', axis=1)
#     data=data.drop('KT-'+str(i)+'-回风温度3的min', axis=1)
#     data = data.drop('KT-' + str(i) + '-回风温度4的max', axis=1)
#     data = data.drop('KT-' + str(i) + '-回风温度4的min', axis=1)
#     data = data.drop('KT-' + str(i) + '-送风温度1的max', axis=1)
#     data = data.drop('KT-' + str(i) + '-送风温度1的min', axis=1)
#     data = data.drop('KT-' + str(i) + '-送风温度4的max', axis=1)
#     data = data.drop('KT-' + str(i) + '-送风温度4的min', axis=1)
#     data.rename(columns={'KT-'+str(i)+'-回风温度1的avg':'KT-'+str(i)+'-回风温度1',
#                          'KT-' + str(i) + '-回风温度2的avg': 'KT-' + str(i) + '-回风温度2',
#                          'KT-' + str(i) + '-回风温度3的avg': 'KT-' + str(i) + '-回风温度3',
#                          'KT-' + str(i) + '-回风温度4的avg': 'KT-' + str(i) + '-回风温度4',
#
#                          'KT-' + str(i) + '-送风温度1的avg': 'KT-' + str(i) + '-送风温度1',
#                          'KT-' + str(i) + '-送风温度4的avg': 'KT-' + str(i) + '-送风温度4'
#                          }, inplace=True)
# data.to_csv(csv_file,index=False)