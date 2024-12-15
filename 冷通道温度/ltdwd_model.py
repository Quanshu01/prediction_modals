import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import warnings
from joblib import dump, load
warnings.filterwarnings("ignore", category=Warning)


plt.rcParams['axes.unicode_minus'] = False
tf.random.set_seed(42)
np.random.seed(42)


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def scaler_x_y(train_X,test_X,train_y,test_y,scaler_x_file,scaler_y_file):

    scale_X = MinMaxScaler()
    scale_y = MinMaxScaler()

    scale_train_X = scale_X.fit_transform(train_X)
    scale_test_X = scale_X.transform(test_X)
    scale_train_y = scale_y.fit_transform(train_y.values.reshape(train_y.shape[0], -1))
    scale_test_y = scale_y.transform(test_y.values.reshape(test_y.shape[0], -1))


    dump(scale_X, scaler_x_file, compress=True)
    dump(scale_y, scaler_y_file, compress=True)
    return scale_train_X, scale_test_X, scale_train_y, scale_test_y, scale_y


def split_train_test(X, y):  # 需要归一化的分割数据集
    test_split = round(len(X) * 0.1)  # 0.1 测试集占训练集比例
    train_X = X[:-test_split]
    test_X = X[-test_split:]
    train_y = y[:-test_split]
    test_y = y[-test_split:]

    return train_X,test_X,train_y,test_y

def split_train_test_year(train_X, train_y,test_X,test_y,scaler_x_file,scaler_y_file):  # 需要归一化的分割数据集

    train_X = pd.read_csv(train_X)
    train_y = pd.read_csv(train_y)
    test_X = pd.read_csv(test_X)
    test_y = pd.read_csv(test_y)

    scale_X = StandardScaler()
    scale_y = StandardScaler()
    scale_train_X = scale_X.fit_transform(train_X)
    scale_test_X = scale_X.transform(test_X)
    scale_train_y = scale_y.fit_transform(train_y.values.reshape(train_y.shape[0], -1))
    scale_test_y = scale_y.transform(test_y.values.reshape(test_y.shape[0], -1))

    # print('scale_X info: \n scaler.mean_:\n {},\n scale_X.var : {}'.format(scale_X.mean_, scale_X.var_))
    # print('\nscale_y info: \n scaler.mean_:\n {},\n scale_y.var : {}'.format(scale_y.mean_, scale_y.var_))
    #
    # print("test_y\n", test_y)
    dump(scale_X, scaler_x_file,compress=True)
    dump(scale_y, scaler_y_file,compress=True)
    return scale_train_X, scale_test_X, scale_train_y, scale_test_y, scale_X, scale_y


"""### 模型"""


# NN
def nn_model(train_X):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(),
                                input_shape=(train_X.shape[1],),name='layer1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ELU())


    model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(),name='layer2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ELU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(),name='layer3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ELU())

    model.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(),name='layer4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #指数衰减学习率
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.96)

    optimizer=tf.keras.optimizers.Adam(exponential_decay)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    return model


# CNN
def cnn_model(train_X):
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)

    model = tf.keras.models.Sequential()
    # 第一层输入卷积，loss曲线收敛不震荡
    model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(train_X.shape[1:])))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #model.summary()
    return model


def gru_model(train_X):
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    input_shape = (train_X.shape[1], train_X.shape[2])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(64, activation='tanh', return_sequences=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  input_shape=input_shape))
    model.add(tf.keras.layers.GRU(128, activation='tanh', return_sequences=True,
                                  kernel_regularizer=tf.keras.regularizers.l2()))
    # model.add(tf.keras.layers.GRU(64, activation='tanh', return_sequences=True,
    #                               kernel_regularizer=tf.keras.regularizers.l2()))

    # model.add(tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # , activation='sigmoid')),
    model.add(tf.keras.layers.Dropout(0.3)),
    model.add(tf.keras.layers.Dense(64, activation='relu'))  # , activation='sigmoid')),
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   # model.summary()  # input shape参数放在 Bidirectional下，而不是 GRU下，才可以正确得到summary
    return model


def train_model(model, train_X, train_y, test_X, test_y, scale_y,h5file, target: str, maxORmin: str, JF_name: str,
                FWQ_name: str,model_name, next_num, input_epoch=100):

    checkpoint = ModelCheckpoint(h5file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    if(model_name=="CNN" or model_name=="GRU"):
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=input_epoch, batch_size=1024,
                        callbacks=callbacks_list, verbose=1)

    #draw(history, test_X, test_y, h5file, target, maxORmin, JF_name, FWQ_name,model_name)
    draw(history, test_X, test_y, scale_y, h5file, target, maxORmin, JF_name, FWQ_name,model_name,next_num)


def draw(history, test_X, test_y,scale_y, h5file, target: str, maxORmin: str, JF_name: str, FWQ_name: str,model_name,next_num):
    model = tf.keras.models.load_model(h5file)  # 导出模型之后看误差图
    #
    # sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer1').output)
    # print('第1层',sub_model.predict(test_X))
    #
    # sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer2').output)
    # print('第2层', sub_model.predict(test_X))
    #
    # sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer3').output)
    # print('第3层', sub_model.predict(test_X))
    #
    # sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer4').output)
    # print('第4层', sub_model.predict(test_X))

    test_pred = model.predict(test_X)
    print(test_pred)

    print("test_y.shape", test_y.shape)
    print("test_pred.shape", test_pred.shape)
    # testY,test_pred反归一化
    test_y = scale_y.inverse_transform(test_y)
    test_pred = scale_y.inverse_transform(test_pred)

    # 粘结结果
    c = np.concatenate([test_y, test_pred], axis=1)
    c = pd.DataFrame(c)
    c = pd.concat([c, abs(c[0] - c[1])], axis=1)
    np.savetxt(fname="data2.csv", X=c, fmt='%.2e', delimiter=',', encoding='utf-8')
    #print(c)
    # print("test_y[:-20]\n", test_y[:-20])
    # print("\ntest_pred[:-20]\n", test_pred[:-20])

    # 计算测试集指标
    r2_result = r2_score(test_y, test_pred)
    rmse_result = np.sqrt(mean_squared_error(test_y, test_pred))
    print("r2:{:.4f},rmse:{:.4f}".format(r2_result, rmse_result))

    # 看收敛与否，绘制mse图
    epochs = range(len(history.history['loss']))
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'b', label='Training mse')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_mse')
    plt.title(target + maxORmin + '_' + JF_name + "JF " + FWQ_name + 'fwq_' +model_name  +str(next_num)+' Traing and Validation mse')
    plt.legend()
    plt.savefig('./pic/'+target + maxORmin + '_' + JF_name + "JF " + FWQ_name + 'fwq_' +model_name  +str(next_num)+' Traing and Validation mse.png')
    #plt.show()

    # 只画测试集：预测-真实
    y = test_y
    # plt.figure()
    # 2022.11.21 修改画布大小，使得测试点更清晰
    plt.figure(figsize=(26, 3), dpi=100)  # 设置画布大小
    num = len(test_pred)  # 2022.11.22 测试画test_pred后num个点
    # Data True
    plt.plot([x for x in range(len(test_pred) - num, len(test_pred))], y[len(y) - num:len(y)], linewidth=1, color='b',
             label='True')
    # Data in Test
    plt.plot([x for x in range(len(test_pred) - num, len(test_pred))], test_pred[len(test_pred) - num:len(test_pred)],
             linewidth=0.7, color='r', label='Pred')
    plt.title(target + maxORmin + '_' + JF_name + "JF " + FWQ_name + 'fwq_' + model_name+ str(next_num) +' test'+' r2:'+str(round(r2_result,4))+' rmse:'+str(round(rmse_result,4)))
    plt.legend(loc='best')
    plt.savefig('./pic/'+target + maxORmin + '_' + JF_name + "JF " + FWQ_name + 'fwq_' + model_name+ str(next_num) +' test.png')
    #plt.show()


def test(test_X, test_y, h5file, target: str,maxORavg, JF_name: str,
                FWQ_name: str,model_name,scaler_x_file,scaler_y_file,next_num):
    model = tf.keras.models.load_model(h5file)  # 导出模型之后看误差图
    scaler_x = load(scaler_x_file)
    scaler_y = load(scaler_y_file)
    test_pred = model.predict(scaler_x.transform(test_X))

    print("test_y.shape", test_y.shape)
    print("test_pred.shape", test_pred.shape)
    # testY,test_pred反归一化
    #test_y = scale_y.inverse_transform(test_y)
    test_pred = scaler_y.inverse_transform(test_pred)

    # 计算测试集指标
    r2_result = r2_score(test_y, test_pred)
    rmse_result = np.sqrt(mean_squared_error(test_y, test_pred))
    print("r2:{:.4f},rmse:{:.4f}".format(r2_result, rmse_result))


    # 只画测试集：预测-真实
    y = test_y
    # plt.figure()
    # 2022.11.21 修改画5布大小，使得测试点更清晰
    plt.figure(figsize=(26, 3), dpi=100)  # 设置画布大小
    num = len(test_pred)  # 2022.11.22 测试画test_pred后num个点
    # Data True
    plt.plot([x for x in range(len(test_pred) - num, len(test_pred))], y[len(y) - num:len(y)], linewidth=1, color='b',
             label='True')
    # Data in Test
    plt.plot([x for x in range(len(test_pred) - num, len(test_pred))], test_pred[len(test_pred) - num:len(test_pred)],
             linewidth=0.7, color='r', label='Test')
    plt.title(target + maxORavg+'_' + JF_name + "JF " + 'KT_' + str(FWQ_name) + '_' + model_name +str(next_num)+ ' test' + ' r2:' + str(
        round(r2_result, 4)) + ' rmse:' + str(round(rmse_result, 4)))
    plt.legend(loc='best')
    plt.savefig('./pic/test2_27_'+target + maxORavg + '_' + JF_name + "JF " + FWQ_name + 'fwq_' + model_name+str(next_num)+' test.png')
    #plt.show()