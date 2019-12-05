
#%% 
# 
######################### import the bitcoin data ############################
import os
dirpath = os.getcwd() # print("current directory is : " + dirpath)
path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\DM_FinalProject'
filepath = os.path.join( dirpath, path2add ,'BTC_USD.csv')
import numpy as np
import pandas as pd
import csv

bitcoin = pd.read_csv(filepath,index_col=1)


print(bitcoin.head())

print(bitcoin.describe())

print(bitcoin.dtypes)
# %% make histogram plot
# import matplotlib
# import matplotlib.pyplot as plt

# #bitcoin['Closing Price (USD)','Date'].plot(legend=True, figsize=(10, 5), title='Bitcoin Price', label='Closing Price (USD)')

# bitcoin.plot( y = 'Closing Price (USD)', legend=True, figsize=(10, 5), title='Bitcoin Daily Price', label = 'Bitcoin')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.savefig('bitcoinprice.png')


# %% 
######################## import S&P500 stock price ##########################3

path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\DM_FinalProject'
filepath = os.path.join( dirpath, path2add ,'GSPC.csv')
stockindex = pd.read_csv(filepath,index_col=0)

print(stockindex.head(10))

print(stockindex.dtypes)

# %% 

# Subtseting the date with price and calculate the change %
BTC = bitcoin[['Closing Price (USD)']]
BTCchange = BTC.pct_change()

# Subtseting the date with price and calculate the change %
SP500 = stockindex[['Close']]
SPchange = SP500.pct_change()

# %% 
##################### combine bitcoin and stock price into one plot ####################333

import matplotlib
import matplotlib.pyplot as plt

# ax = plt.gca()

# bitcoin.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# stockindex.plot(kind='line', y = 'Close', legend=True, figsize=(10, 5), label = 'S&P 500', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.show()


# # %%
# plt.figure(figsize=(10,8))
# top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
# bottom = plt.subplot2grid((4,4), (3,0), rowspan=3, colspan=4)
# top.plot(BTC.index, BTC['Closing Price (USD)']) #CMT.index gives the dates
# bottom.plot(SP500.index, SP500['Close']) 
 
# # set the labels
# top.axes.get_xaxis().set_visible(False)
# top.set_title('Bitcoin Data')
# top.set_ylabel('Closing Price (USD)')
# bottom.set_title('S&P 500 Data')
# bottom.set_ylabel('Close')
# ax = plt.gca()
# ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
#ax.tick_params(pad=20)

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot()

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = ax1.twinx()

# BTC.plot( y='Closing Price (USD)', ax=ax1, figsize=(10, 8), legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs S&P 500")
# ax1.set_ylabel('Closing Price (USD)')
# ax1.legend(loc=1)

# SP500.plot( y='Close', kind='line', ax=ax2, figsize=(10, 8), label = 'S&P 500', color = 'red')
# ax2.yaxis.set_label_text("")
# ax2.set_ylabel('Closing Price (USD)')
# ax2.legend(loc=2)

#%%

fig = plt.figure(figsize=(15, 10))
# Divide the figure into a 2x1 grid, and give me the first section
ax1=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

BTC.plot( y='Closing Price (USD)', ax=ax1, legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs S&P 500")
ax1.set_ylabel('Closing Price (USD)')
ax1.legend(loc=1)
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')

SP500.plot( y='Close', kind='line', ax=ax2, label = 'S&P 500', color = 'red')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right')
ax2.set_xlabel('Date', color="red") 
ax2.set_ylabel('Closing Price (USD)', color='red')  
ax2.legend(loc=2)
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")


#%%
################ BTC and S&P 500 daily change percentage #######################


# ax = plt.gca()
# BTCchange.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# SPchange.plot(kind='line', y = 'Close', legend=True, figsize=(10, 5), label = 'S&P 500', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Percentage(%)')
# plt.show()

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTCchange.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(15, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Price Change %')

# SPchange.plot( y='Close', kind='line', ax=ax2, figsize=(15, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("S&P 500 Data")
# ax2.set_ylabel('Price Change %')
# fig.subplots_adjust(hspace=0.3)

# %% 

############################ import gold index ###########################
path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\DM_FinalProject'
filepath = os.path.join( dirpath, path2add ,'WGC-GOLD_DAILY_USD.csv')
goldindex= pd.read_csv(filepath, index_col=0)

goldindex.head()
goldindex.dtypes

goldchange = goldindex.pct_change()

#%%

# ############### BTC and Gold daily Price #######################

# import matplotlib
# import matplotlib.pyplot as plt
# ax = plt.gca()

# bitcoin.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# goldindex.plot(kind='line', y = 'Value', legend=True, figsize=(10, 5), label = 'Gold', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.show()

# ax = plt.gca()
# BTCchange.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# goldchange.plot(kind='line', y = 'Value', legend=True, figsize=(10, 5), label = 'Gold', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Percentage(%)')
# plt.show()

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTC.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(10, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Closing Price (USD)')

# goldindex.plot( y='Value', kind='line', ax=ax2, figsize=(10, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("Gold Data")
# ax2.set_ylabel('Value (USD)')
# fig.subplots_adjust(hspace=0.3)

# #%%

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot()

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = ax1.twinx()

# BTC.plot( y='Closing Price (USD)', ax=ax1, figsize=(10, 8), legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs Gold Index")
# ax1.set_ylabel('Closing Price (USD)')
# ax1.legend(loc=1)

# goldindex.plot( y='Value', kind='line', ax=ax2, figsize=(10, 8), label = 'Gold', color = 'red')
# ax2.yaxis.set_label_text("")
# ax2.set_ylabel('Value (USD)')
# ax2.legend(loc=2)

#%%

fig=plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

BTC.plot( y='Closing Price (USD)', ax=ax1, legend=True, label = 'BTC')
ax1.xaxis.set_label_text("")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price (USD)")
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')
ax1.legend(loc=1)

goldindex.plot( y='Value', kind='line', ax=ax2, label = 'Gold', color = 'red')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.yaxis.set_label_text("")
ax2.set_xlabel('Date', color="red") 
ax2.set_ylabel('Closing Price (USD)', color='red')  
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")
ax2.legend(loc=2)

#%%
################ BTC and Gold daily change percentage #######################

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTCchange.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(15, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Price Change %')

# goldchange.plot( y='Value', kind='line', ax=ax2, figsize=(15, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("Gold Data")
# ax2.set_ylabel('Value Change %')
# fig.subplots_adjust(hspace=0.3)

# #%%
# from statsmodels.graphics.tsaplots import plot_acf
# from matplotlib import pyplot

# plot_acf(BTC)
# pyplot.show()

# #%%

# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(BTC, lags=100)
# pyplot.show()
#%%
################ Try to find correlation#######################

from statsmodels.formula.api import ols

BTCSP = pd.merge(BTC, SP500, on='Date')
BTCSP = BTCSP.rename(columns={"Closing Price (USD)": "btc", "Close": "sp500"}).dropna()

modelBTCSP = ols(formula='btc ~ sp500', data=BTCSP).fit()
print( modelBTCSP.summary() )

#%%

##### gold
BTCGOLD = pd.merge(BTC, goldindex, on='Date')
BTCGOLD = BTCGOLD.rename(columns={"Closing Price (USD)": "btc", "Value": "gold"}).dropna()

modelBTCGOLD = ols(formula='btc ~ gold', data=BTCGOLD).fit()
print( modelBTCGOLD.summary() )


# %%
################ Try to find correlation of change %#######################

#####sp500
from statsmodels.formula.api import ols
BTCSPchange = pd.merge(BTCchange, SPchange, on='Date')
BTCSPchange = BTCSPchange.rename(columns={"Closing Price (USD)": "btc", "Close": "sp500"}).dropna()

modelBTCSP = ols(formula='btc ~ sp500', data=BTCSPchange).fit()
print( modelBTCSP.summary() )

##### gold
BTCGOLDchange = pd.merge(BTCchange, goldchange, on='Date')
BTCGOLDchange = BTCGOLDchange.rename(columns={"Closing Price (USD)": "btc", "Value": "gold"}).dropna()

modelBTCGOLD = ols(formula='btc ~ gold', data=BTCGOLDchange).fit()
print( modelBTCGOLD.summary() )

#%%

# import os
# dirpath = os.getcwd() # print("current directory is : " + dirpath)
# path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\DM_FinalProject'
# filepath = os.path.join( dirpath, path2add ,'BTC_USD.csv')
# import numpy as np
# import pandas as pd
# import csv

# bitcoin = pd.read_csv(filepath)


# print(bitcoin.head())

# print(bitcoin.describe())

# print(bitcoin.dtypes)



#%%
forecast_out = 5 #'n=30' days
BTC['Prediction'] = BTC[['Closing Price (USD)']].shift(-forecast_out)
print(BTC.tail(10))

#%%

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(BTC.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)


y = np.array(BTC['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# %%
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

#%%

x_forecast = np.array(BTC.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)# %%


# %%

# Print linear regression model predictions for the next '30' days
lr_prediction = lr.predict(x_forecast)
print(np.asarray(lr_prediction).reshape(5,1))
# Print support vector regressor model predictions for the next '30' days
# svm_prediction = svr_rbf.predict(x_forecast)
# print(np.asarray(svm_prediction).reshape(5,1))


# %%
#https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a