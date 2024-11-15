import pandas as pd
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fwq='AB'
data=pd.read_csv('./csv_data/X/202_'+fwq+'_fwqgl_X_10.csv')
plt.figure(figsize=(20,10),dpi=300)
#plt.plot([i for i in range(len(data))],data['hours'])
plt.plot([str(i)+str(data['hours'][i]) for i in range(len(data))],data.iloc[:,0])
plt.title(fwq+'服务器功率')
plt.savefig('./csv_data/'+fwq+'服务器功率.png')
plt.show()

# fig, ax1 = plt.subplots(figsize=(20,10))
# # plt.figure(figsize=(20,10),dpi=300)
# ax2 = ax1.twinx()  # 做镜像处理
# ax2.plot([i for i in range(len(data))],data['hours'],'b')
# ax1.plot([i for i in range(len(data))], data.iloc[:,0],'g')
#
#
# ax1.set_xlabel('X data')  # 设置x轴标题
# ax1.set_ylabel(fwq+'服务器功率', color='g')  # 设置Y1轴标题
# ax2.set_ylabel('小时', color='b')  # 设置Y2轴标题
# plt.savefig('./csv_data/'+fwq+'服务器功率.png')