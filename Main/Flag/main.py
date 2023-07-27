import MetaTrader5 as mt5 
import pandas as pd 
import numpy as np
from collections import deque
from datetime import datetime
import os
import matplotlib.pyplot as plt 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics  import f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


NORMAL_VALUE = 19
ACCOUNT = input() # your account  
PASSWORD = int(input())  # your password
TICKER_NAME = "AUDCAD"
SERVERNAME = "Alpari-MT5-Demo"
TIME_START = datetime(2022, 11, 20)
TIME_END = 1690482014


pd.options.mode.chained_assignment = None

if not mt5.initialize():
    print(f"The Error is ,{mt5.last_error()} \n")

login_account = mt5.login(ACCOUNT, password=PASSWORD, server=SERVERNAME)
if login_account:
    print(f"Succesfully connected... version's {mt5.__version__}") # my version is 5.0.37
else:
    print(f"Connection failed. Error, {mt5.last_error()}")


df = pd.DataFrame(mt5.copy_rates_range(TICKER_NAME, 
                                      mt5.TIMEFRAME_H1,
                                      TIME_START,
                                      TIME_END)
                 )  

df.drop(["tick_volume", "real_volume", 
         "spread", "high", "low", "time"], axis=1, inplace=True)



def ReadPositiveNegetive(file_name):
    with open(file_name, 'r') as read:
        lst = read.read()[1:-1].replace('[', '').replace(']', '').split(',')

    indexes = []
    cnt = 0
    for i in lst:    
        indexes.append(int(i.strip()[1:-1]))


    indexes = sorted(indexes)[2:]
    tow_index = []
    for i in range(0, len(indexes), 2):
        tow_index.append(indexes[i:i+2])
    
    return len(tow_index), tow_index


def ReadNeutral(file_name):
    with open(file_name, 'r') as read:
        lst = read.read()[1:-1].replace('[', '').replace(']', '').split(',')
        
    indexes = []
    for i in lst:    
        indexes.append([int(i.strip())-NORMAL_VALUE, int(i.strip())])

    return len(sorted(indexes)[1:]), sorted(indexes)[1:]
    

# New We must append index of data to main_df dataframe
def DataSetTogether(df, index, target):
    all_data = []
    for ind in index:
        data = []
        for val in df.loc[ind[0]:ind[1]-1].values:
            data.append(val[0])
            data.append(val[1])
        
        if len(data)==38:
            data.append(target)
        
        if 40>len(data)>36:
            all_data.append(data)
        
    return all_data


def ForcastTest(df_test):
    indx_buy = []
    buy = []
    indx_sell = []
    sell = []
    for indx in range(0, len(df_test), NORMAL_VALUE):
        test = []
        for val in df_test.iloc[indx:indx+NORMAL_VALUE].values:
            test.append(val[0])
            test.append(val[1])
        
        try:
            test = scaler.fit_transform(np.array(test).reshape(-1, 1))
            forcast = model.predict(test.reshape(1, -1))
            if forcast==1:
                df_test['flag'].iloc[indx] = forcast
                buy.append(df_test['close'].iloc[indx+2])
                indx_buy.append(indx+2)
                
            elif forcast==-1:
                df_test['flag'].iloc[indx] = forcast
                sell.append(df_test['close'].iloc[indx+2])
                indx_sell.append(indx+2)
                
        except Exception as e:
            print(e)

    return buy, indx_buy, sell, indx_sell


def chandlesPlot(d, o, c):
    plt.figure(figsize=(24, 14))
    color = []
    for open_p, close_p in zip(o, c):
        if open_p<close_p:
            color.append("g")
        else:
            color.append("r")

    plt.bar(d, height=np.abs(o-c), 
            width=0.8, 
            color=color, 
            bottom=np.min((o, c), axis=0))


def ShowData(main_df, saving=False):
    chandlesPlot(main_df.index, 
                 main_df["open"], 
                 main_df["close"])
    
    plt.scatter(indx_buy, buy, color='g', linewidths=15, label='Buy')
    plt.scatter(indx_sell, sell, color='r', linewidths=15, label='Sell') 
    plt.title(TICKER_NAME)
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend(loc="lower left")
    if saving:
        if os.path.exists("img_test"):
            plt.savefig(f"img_test/Simple_Save.jpg")
        else:
            os.mkdir("img_test")
            plt.savefig(f"img_test/Simple_Save.jpg") 

ind_neutral, index_netral = ReadNeutral('DataSetNeutal.txt')
ind_negetive, index_negetive = ReadPositiveNegetive('DataSetNegetive.txt')
ind_positive, index_positive = ReadPositiveNegetive('DataSetPositive.txt')


neg_data = DataSetTogether(df, index_negetive, -1)
neu_data = DataSetTogether(df, index_netral, 0)
pos_data = DataSetTogether(df, index_positive, 1)

dataset = [neg_data, neu_data, pos_data]


X = []
y = []
for j in dataset:
    for i in j:
        X.append(i[:-1])
        y.append(i[-1])
        
X, y = np.array(X), np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=2)
print(f"Shape of X_train {X_train.shape}")
print(f"Shape of the Y_train {y_train.shape}")
print(f"Shape of the X_valid {X_test.shape}")
print(f"Shape of the Y_valid {y_test.shape}")


model = RandomForestClassifier(n_estimators=150, n_jobs=-1)
model.fit(X, y)

pred = model.predict(X_test)
 
print(accuracy_score(y_test, pred) * 100)
print(mean_absolute_error(y_test, pred))
print(mean_absolute_error(y_test, pred))


df_test = pd.DataFrame(mt5.copy_rates_range('EURUSD', 
                                      mt5.TIMEFRAME_H1,
                                      datetime(2023, 7, 10),
                                      TIME_END)
                 )  

df_test.drop(["tick_volume", "real_volume", 
         "spread", "high", "low", "time"], axis=1, inplace=True)

df_test['flag'] = np.NaN


buy, indx_buy, sell, indx_sell = ForcastTest(df_test)

ShowData(df_test, saving=True)

