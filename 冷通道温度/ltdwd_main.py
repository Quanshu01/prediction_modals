import os

import ltdwd_export_csv_data_loader as dl
#from ltdwd_export_csv_data_process import Logger
import sys
from ltdwd_export_csv_data_process import FWQ_202,FWQ_203,KT_toFwq_list_203,KT_toFwq_list_202
from ltdwd_model import Logger
import sys
import ltdwd_model as model

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=Warning)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



def ltdwd_export_data(swwd_falg):
    # log/h5 命名: JF202_ABfwq_ltdwdAVG_next15
    all_file_name = "JF" + jf + "_fwq_" + fwq +"_"+ target + '_' + maxORavg + "_next" + str(next_num)
    # 打印预测需要信息
    # You are going to predict JF202_ABfwq_ltdwdAVG_next15
    print("You are going to predict " + all_file_name)
    # 生成训练数据，方便之后训练时间集中在模型本身，而不是加载数据
    X, y = dl.deal_odd_data(maxORavg=maxORavg, JF_num=jf, FWQ_name=fwq, next_num=next_num,past_num=past_num,train_X_data_csv=train_X_data_csv,train_y_data_csv=train_y_data_csv,swwd_flag=swwd_falg)
    return X, y


def ltdwd_train():
    # log/h5 命名
    all_file_name ="JF" + jf + "_fwq_" + fwq +"_"+ target + '_' + maxORavg +  '_' + model_name

    # 打印在控制台的信息 保存至 log文件
    log_export_path_name = './log/'
    logfile = log_export_path_name + "log_" + all_file_name + ".log"  # 打印 并保存到 log文件
    sys.stdout = Logger(filename=logfile, stream=sys.stdout)  # 将控制台输出保存到文件中
    print('以下所有在控制台打印的内容，将会保存到Log文件中')

    # 打印预测需要信息
    # You are going to predict JF205_ABfwq_ltdwd_max_past15only_GRU
    print("You are going to predict " + all_file_name)

    # 2022.12.4 去当前时刻变量，仅保留当前时刻变量
    # /Users/kwok/Desktop/AI_for_env/data/train/y/ltdwd_max/past15_only/train_olddata_past15_202_AB_ltdwdmax_y.csv


    X = pd.read_csv(train_X_data_csv)
    y = pd.read_csv(train_y_data_csv)
    print("chosen JF" + jf + fwq + "fwq train data:\n" + "  X's shape:", X.shape, "\n  y's shape:", y.shape)
    h5_export_path_name = './h5/'
    h5file = h5_export_path_name + all_file_name + ".h5"  # 神经网络模型 h5文件 命名
    # 划分数据集（训练/测试）（归一化/不归一化）
    scaler_x_file='./scaler/X/'+all_file_name+"_X.bin"
    scaler_y_file='./scaler/y/'+all_file_name+"_y.bin"
    train_X,test_X,train_y,test_y = model.split_train_test(X, y)
    train_X,test_X,train_y,test_y, scale_y=model.scaler_x_y(train_X,test_X,train_y,test_y,scaler_x_file,scaler_y_file)
    # train_X=X
    # train_y=y
    ''' ----------------------------多种模型训练 ---------------------------'''
    # NN
    if model_name == 'NN':
        m = model.nn_model(train_X)
        input_epoch = 200
    # CNN
    elif model_name == 'CNN':
        m = model.cnn_model(train_X)
        input_epoch = 100
    # GRU
    elif model_name == 'GRU':
        m = model.gru_model(train_X)
        input_epoch = 200


    model.train_model(m, train_X, train_y, test_X, test_y, scale_y, h5file, target, maxORavg, jf, fwq,model_name, input_epoch)
    # model.train_model(m, train_X, train_y, test_X, test_y, h5file, target, maxORavg, jf, fwq,model_name, input_epoch)
    # model.test(train_X, train_y, h5file, target,maxORavg, jf,
    #             fwq,model_name,scaler_x_file,scaler_y_file)
    print("You are going to predict " + all_file_name)

def ltdwd_test():
    # log/h5 命名
    all_file_name ="JF" + jf + "_fwq_" + fwq +"_"+ target + '_' + maxORavg +  '_' + model_name

    # 打印预测需要信息
    # You are going to predict JF205_ABfwq_ltdwd_max_past15only_GRU
    print("You are going to test " + all_file_name)
    X = pd.read_csv(train_X_data_csv)
    y = pd.read_csv(train_y_data_csv)
    print("chosen JF" + jf + fwq + "fwq train data:\n" + "  X's shape:", X.shape, "\n  y's shape:", y.shape)
    h5_export_path_name = './h5/'
    h5file = h5_export_path_name + all_file_name + ".h5"  # 神经网络模型 h5文件 命名
    # 划分数据集（训练/测试）（归一化/不归一化）
    scaler_x_file='./scaler/X/'+all_file_name+"_X.bin"
    scaler_y_file='./scaler/y/'+all_file_name+"_y.bin"

    train_X=X
    train_y=y
    ''' ----------------------------多种模型训练 ---------------------------'''
    # NN
    if model_name == 'NN':
        m = model.nn_model(train_X)
        input_epoch = 200
    # CNN
    elif model_name == 'CNN':
        m = model.cnn_model(train_X)
        input_epoch = 100
    # GRU
    elif model_name == 'GRU':
        m = model.gru_model(train_X)
        input_epoch = 200


    model.test(train_X, train_y, h5file, target,maxORavg, jf,
                fwq,model_name,scaler_x_file,scaler_y_file)
    print("You are going to predict " + all_file_name)

if __name__ == '__main__':
    '''
            超参数
    '''
    target = 'ltdwd'
    for jf in ['202']:
        #fwq = 'NP'  # AB-0,CD-1,EF-2,GH-3,JK-4,LM-5,NP-6
        for maxORavg in ['max',]:
            next_num = 10  # 未来next_num时刻（未来1时刻、未来5时刻、未来15时刻）
            past_num=15
            model_name = "NN"  # 模型名: NN/CNN/GRU
            if (jf == '202'):
                fwq_list = FWQ_202
            elif (jf == "203"):
                fwq_list = FWQ_203
            for fwq in fwq_list:
                if(not (jf=='203' and fwq in ['EF','GH'])):
                    #if(fwq=='JK'):
                        train_X_data_csv = './csv_data/X/ltdwd_' + maxORavg + '/' + jf + '_' + fwq + '_ltdwd' + maxORavg + '_X_'+str(next_num) + '.csv'
                        train_y_data_csv = './csv_data/y/ltdwd_' + maxORavg + '/' + jf + '_' + fwq + '_ltdwd' + maxORavg + '_y_'+str(next_num) + '.csv'
                        #lJJtdwd_export_data()
                        #if(not os.path.exists(train_X_data_csv) and not os.path.exists(train_y_data_csv)):
                        ltdwd_export_data(swwd_falg=True)
                        #ltdwd_train()
                        ltdwd_test()








