import os

import sfwd_export_csv_data_loader as dl
#from sfwd_export_csv_data_process import Logger
import sys
from sfwd_model import Logger
import sys
import sfwd_model as model

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


def sfwd_export_data(jf, kt, target, next_num, past_num,swwd_flag,train_X_data_csv,train_y_data_csv,year):
    all_file_name = "JF" + jf + "_" + str(kt) + "kt_" + target +"_next" + str(next_num)+"_"+model_name
    # 打印预测需要信息
    print("You are going to predict " + all_file_name)
    # 生成训练数据，方便之后训练时间集中在模型本身，而不是加载数据
    X, y =dl.deal_odd_data(JF_num=jf, KT_name=kt, next_num=next_num,past_num=past_num,train_X_data_csv=train_X_data_csv,train_y_data_csv=train_y_data_csv,swwd_flag=swwd_flag,year=year)
    return X, y


def sfwd_train(jf, kt, target, model_name, next_num, past_num, train_X_data_csv, train_y_data_csv, test_X_data_csv=0,test_y_data_csv=0):
    # log/h5 命名
    all_file_name = "JF" + jf + "_" + "kt_" +str(kt) + target +'_' +  str(next_num)+"_"+model_name

    # 打印在控制台的信息 保存至 log文件
    log_export_path_name = './log/'
    logfile = log_export_path_name + "log_" + all_file_name + ".log"  # 打印 并保存到 log文件
    sys.stdout = Logger(filename=logfile, stream=sys.stdout)  # 将控制台输出保存到文件中
    print('以下所有在控制台打印的内容，将会保存到Log文件中')

    # 打印预测需要信息
    print("You are going to predict " + all_file_name)
    X = pd.read_csv(train_X_data_csv)
    y = pd.read_csv(train_X_data_csv)
    print("chosen JF" + jf + str(kt) + "kt train data:\n" + "  X's shape:", X.shape, "\n  y's shape:", y.shape)
    h5_export_path_name = './h5/'
    h5file = h5_export_path_name + all_file_name + ".h5"  # 神经网络模型 h5文件 命名
    # 划分数据集（训练/测试）（归一化/不归一化）
    scaler_x_file='./scaler/X/'+all_file_name+"_X.bin"
    scaler_y_file='./scaler/y/'+all_file_name+"_y.bin"
    if test_X_data_csv == 0:
        train_X, test_X, train_y, test_y, scale_X, scale_y = model.split_train_test(X, y, scaler_x_file, scaler_y_file)
    else:
        train_X, test_X, train_y, test_y, scale_X, scale_y = model.split_train_test_year(train_X=train_X_data_csv,
                                                                                    train_y=train_y_data_csv,
                                                                                    test_X=test_X_data_csv,
                                                                                   test_y=test_y_data_csv,
                                                                                   scaler_x_file=scaler_x_file,
                                                                                   scaler_y_file=scaler_y_file)

    #train_X, test_X, train_y, test_y, scale_X, scale_y = model.split_train_test(X, y,scaler_x_file,scaler_y_file)
    # train_X=X
    # train_y=y
    ''' ----------------------------多种模型训练 ---------------------------'''
    # NN
    if model_name == 'NN':
        m = model.nn_model(train_X)
        input_epoch = 100
    # CNN
    elif model_name == 'CNN':
        m = model.cnn_model(train_X)
        input_epoch = 100
    # GRU
    elif model_name == 'GRU':
        m = model.gru_model(train_X)
        input_epoch = 100
    # KAN
    if model_name == "KAN":
        m = model.kan_model(train_X)
        input_epoch = 100
    #model.test(train_X, train_y, h5file, target, jf, kt, model_name,scaler_x_file,scaler_y_file)

    model.train_model1(m, train_X, train_y, test_X, test_y, scale_y, h5file, target,jf, kt,model_name,next_num,input_epoch)

    print("You are going to predict " + all_file_name)

# def sfwd_test():
#     # log/h5 命名
#     all_file_name = "JF" + jf + "_" + "kt_" +str(kt) + target +'_' + model_name

#     # 打印预测需要信息
#     print("You are going to test " + all_file_name)
#     X = pd.read_csv(train_X_data_csv)
#     y = pd.read_csv(train_y_data_csv)
#     print("chosen JF" + jf + str(kt) + "kt train data:\n" + "  X's shape:", X.shape, "\n  y's shape:", y.shape)

#     # 划分数据集（训练/测试）（归一化/不归一化）
#     scaler_x_file='./scaler/X/'+all_file_name+"_X.bin"
#     scaler_y_file='./scaler/y/'+all_file_name+"_y.bin"
#     train_X=X
#     train_y=y

#     ''' ----------------------------多种模型训练 ---------------------------'''
#     # NN
#     if model_name == 'NN':
#         m = model.nn_model(train_X)
#         input_epoch = 200
#     # CNN
#     elif model_name == 'CNN':
#         m = model.cnn_model(train_X)
#         input_epoch = 100
#     # GRU
#     elif model_name == 'GRU':
#         m = model.gru_model(train_X)
#         input_epoch = 200

#     h5_export_path_name = './h5/'
#     h5file = h5_export_path_name + all_file_name + ".h5"  # 神经网络模型 h5文件 命名
#     model.test_model1(train_X, train_y, scaler_x_file,scaler_y_file, h5file, target, jf, kt,model_name)
#     print("You are going to test " + all_file_name)

if __name__ == '__main__':
    # '''
    #         超参数
    # '''
    # target = 'sfwd'
    # for jf in ['202']:  # 机房号
    #     next_num = 10  # 未来next_num时刻（未来1时刻、未来5时刻、未来15时刻）
    #     past_num = 15  # 未来next_num时刻（未来1时刻、未来5时刻、未来15时刻）
    #     model_name = "NN"  # 模型名: NN/CNN/GRU
    #     if (jf == '202'):
    #         kt_list = [i for i in range(1,21)]
    #     elif (jf == "203"):
    #         kt_list = [i for i in range(1,14)]
    #         kt_list.remove(3)
    #         kt_list.remove(4)
    #         kt_list.remove(5)
    #     for kt in kt_list:
    #         if(kt in [1,2,4,8,10,12]):
    #                 train_X_data_csv = './csv_data/X/' + jf + '_' + str(kt) + '_sfwd_X_'+str(next_num) + '.csv'
    #                 train_y_data_csv = './csv_data/y/' + jf + '_' + str(kt) + '_sfwd_y_' +str(next_num) +'.csv'
    #                 #if(not os.path.exists(train_X_data_csv) and not os.path.exists(train_y_data_csv)):
    #                 sfwd_export_data(swwd_flag=True)
    #                 #sfwd_train()
    #                 sfwd_test()

    target = 'sfwd'
    for jf in ['202']:
        kt_list = [i for i in range(1,21)]
        # 定义next_num的取值列表
        # next_num_list = [15]
        next_num_list = [5, 10, 15]
        # next_num_list = [5, 10, 15]
        # 定义model_name的取值列表
        model_name_list = ['KAN']
        # model_name_list = ['NN', 'GRU']
        past_num = 15

        for model_name in model_name_list:
            for next_num in next_num_list:
                for kt in kt_list:
                    # if(kt in [1]):
                    if(kt in [1,4,10,12]):
                    # if (kt in [1, 2, 4, 8, 10, 12]):
                        train_X_data_csv_2022 = f"./csv_data/X/{jf}_{kt}_sfwd_X_{next_num}_2022.csv"
                        train_y_data_csv_2022 = f"./csv_data/y/{jf}_{kt}_sfwd_y_{next_num}_2022.csv"

                        train_X_data_csv_2023 = f"./csv_data/X/{jf}_{kt}_sfwd_X_{next_num}_2023.csv"
                        train_y_data_csv_2023 = f"./csv_data/y/{jf}_{kt}_sfwd_y_{next_num}_2023.csv"

                        if model_name == 'NN':
                            sfwd_export_data(jf, kt, target, next_num, past_num,swwd_flag=True,train_X_data_csv=train_X_data_csv_2022, train_y_data_csv=train_y_data_csv_2022,year=2022)
                            sfwd_export_data(jf, kt, target, next_num, past_num,swwd_flag=True,train_X_data_csv=train_X_data_csv_2023, train_y_data_csv=train_y_data_csv_2023,year=2023)

                        sfwd_train(jf, kt, target, model_name, next_num, past_num, train_X_data_csv_2022, train_y_data_csv_2022, train_X_data_csv_2023, train_y_data_csv_2023)







