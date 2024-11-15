import pandas as pd
import numpy as np
# data=pd.DataFrame(columns=['a','b'])
# data['a']=[-1,-2,0,2,3,5,6,5,np.nan,15,20,4,np.nan]
# data['b']=[1,2,0,-2,3,-5,np.nan,56,5,15,20,np.nan,1]
# print(data)
# data=data.loc[(data>0).all(axis=1) & (data<20).all(axis=1)]
# print(data)

data=pd.read_csv(r'E:\lry\模型汇总\服务器功率\csv_data\X\202_AB_fwqgl_X_10.csv')
data=data.dropna(axis=0)
data