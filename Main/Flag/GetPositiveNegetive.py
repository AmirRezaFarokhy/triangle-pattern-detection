import MetaTrader5 as mt5 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime

ACCOUNT = input() # your account  
PASSWORD = int(input())  # your password
TICKER_NAME = "AUDCAD"
SERVERNAME = "Alpari-MT5-Demo"

answer = input("You wanna collect postive (positive -> y) (negetive -> n) >>>  ")
if answer=='y':
    POSITIVE = True
elif answer=='n':
    POSITIVE = False
else:
    print("Pleas type y or n...")


pd.options.mode.chained_assignment = None

if not mt5.initialize():
	print(f"The Error is ,{mt5.last_error()}")

login_account = mt5.login(ACCOUNT, password=PASSWORD, server=SERVERNAME)
if login_account:
	print(f"Succesfully connected... version's {mt5.__version__}") # my version is 5.0.37
else:
	print(f"Connection failed. Error, {mt5.last_error()}")
	

df = pd.DataFrame(mt5.copy_rates_range(TICKER_NAME, 
									  mt5.TIMEFRAME_H1,
									  datetime(2022, 11, 20),
									  mt5.symbol_info_tick(TICKER_NAME).time)
				 )  

df.drop(["tick_volume", "real_volume", "spread"], axis=1, inplace=True)

def chandlesPlot(d, o, h, l, c):
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
    plt.bar(d, height=h-l, width=0.3, color=color, bottom=l) 


def ShowData(main_df):
    chandlesPlot(main_df.index, main_df["open"], 
                main_df["high"], main_df["low"], 
                main_df["close"])

chunk = len(df) // 10
important_lst = []
for vis in range(0, len(df), chunk):
    ShowData(df.loc[vis:vis+chunk])
    plt.show()
    lst = []
    how_many = int(input('How many you find?! '))
    for _ in range(how_many * 2):
        index = input('Give me index one >>> ')
        lst.append(index)
    important_lst.append(lst)


print(important_lst)


if POSITIVE:
    with open('DataSetPositive.txt', 'w') as writen:
        writen.write(str(important_lst))
        writen.close()

else:
    with open('DataSetNegetive.txt', 'w') as writen:
        writen.write(str(important_lst))
        writen.close()
