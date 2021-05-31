# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:02:04 2019

@author: Diego Bustillo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:29:09 2019

@author: alessandrosollazzo
"""

"""
Created on Mon Apr 15 15:19:49 2019

@author: alessandrosollazzo
"""
#Import all the libraries and packages needed to build this machine learning algorithm
from pandas_datareader import data as web
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from IPython import get_ipython
import warnings

#Getting the data and making it usable; To create any algorithm we need data to train the algorithm and then to make predictions on new unseen data.
#We will get the data from yahoo.com To do this we will use the data reader function from panda's library. It enables to get data from online data sources
##We are fetching the data of the SPDR ETF linked to S&P 500. This stock can be used as a proxy for the performance of the S&P 500 index,specyfing the year starting from which we will pulling the data.
#Once the data is in, we will discard any data other than the OHLC, such as volume and adjusted Close, to create our data frame ‘df ’.

avg_err={}
avg_train_err={}
df = web.get_data_yahoo('ETFC' ,start='2004-01-01',end='2019-05-01')
df=df[['Open','High','Low','Close','Volume']]
df['open']=df['Open'].shift(1)
df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)
df['close']=df['Close'].shift(1)
df['volume']=df['Volume'].shift(1)

X=df[['open','high','low','close']]
y =df['Close']

#There are some parameter that the machine learning can't learn over but needs to be iterated over. We use them to see which predefined functions or parameters yield the best fit function.
#We have used Lasso regression which uses L1 type of regulariyation, it's used to predict continous data. n this example, we used 5 fold cross validation. In a k-fold cross-validation, the original sample is randomly partitioned into k equal size subsamples

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

steps = [('imputation', imp),
         ('scaler',StandardScaler()),
         ('lasso',Lasso())]        

pipeline =Pipeline(steps)


parameters = {'lasso__alpha':np.arange(0.0001,10,.0001),
              'lasso__max_iter':np.random.uniform(100,100000,4)}
              
              
reg = rcv(pipeline, parameters,cv=5)

#Splitting the data into test and train sets. We passon the OHLC data with one 
#day lag as the data frame X and the Close values of the current days as y.
#Let create a dictionary that holds the size of the train data set and its corresponding average prediction error.
#It's time the get the best-fit parameters to create a new function: we want to measure the performance of the regression
#function as compared to the size of the input database. For this reason we use the loop to iterate over the same data set
#but eith different lengths.
#First we created a set of periodic numbers "t" starting from 50 to 97, in stps of 3. The purpose of thesenumbers is
#to choose the percentage size of the dataset that will be used as train data set.Second, for a given value of ‘t’ I split
#the length of the data set to the nearest integer corresponding to this percentage. Then I divided the total data into train
#data, which includes the data from the beginning till the split, and test data, which includes the data from the split till
#the end. The reason for adopting this approach and not using the random split is to maintain the continuity of the time series.

for t in np.arange(50,97,3):
    get_ipython().magic('reset_selective -f reg1')
    split = int(t*len(X)/100)
    reg.fit(X[:split],y[:split])
    best_alpha = reg.best_params_['lasso__alpha']
    best_iter= reg.best_params_['lasso__max_iter']
    reg1= Lasso(alpha=best_alpha,max_iter=best_iter)
    X=imp.fit_transform(X,y)
    reg1.fit(X[:split],y[:split])
    
  #Making the predictions and checking the performance  
    df['P_C_%i'%t]=0.
    df.iloc[:,df.columns.get_loc('P_C_%i'%t)]=reg1.predict(X[:])
    df['Error_%i'%t]= np.abs(df['P_C_%i'%t]-df['Close'])

    e =np.mean(df['Error_%i'%t][split:])
    train_e= np.mean(df['Error_%i'%t][:split])
    avg_err[t]=e
    avg_train_err[t]=train_e
Range =df['high'][split:]-df['low'][split:]

#now it's time to plot
plt.scatter(list(avg_err.keys()),list(avg_err.values()), label='test_error')
plt.scatter(list(avg_train_err.keys()),list(avg_train_err.values()),label='train_error')
plt.legend(loc='best')
print ('\nAverage Range of the Day:',np.average(Range))

#Our algorithm is doing better in the test data compared to the train data. This observation in itself is a red flag. There are a few reasons why our test data error could be better than the train data error:
#If the train data had a greater volatility (Daily range) compared to the test set, then the prediction would also exhibit greater volatility.
#If there was an inherent trend in the market that helped the algo make better predictions.
#Now, let us check which of these cases is true. If the range of the test data was less than the train data, then the error should have decreased after passing more than 80% of the data as a train set, but it increases.

from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn import mixture as mix
import seaborn as sns 
import matplotlib.pyplot as plt

df= web.get_data_yahoo('ETFC',start= '2004-01-01', end='2019-05-01')
df=df[['Open','High','Low','Close']]
df['open']=df['Open'].shift(1)
df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)
df['close']=df['Close'].shift(1)

df=df[['open','high','low','close']]

df=df.dropna()

#Next, we will instantiate an unsupervised machine learning algorithm using the ‘Gaussian mixture’ model from sklearn.
unsup = mix.GaussianMixture(n_components=4, 
                            covariance_type="spherical", 
                            n_init=100, 
                            random_state=42)

#Next, we will fit the data and predict the regimes. Then we will be storing these regime predictions in a new variable called regime.
unsup.fit(np.reshape(df,(-1,df.shape[1])))
regime = unsup.predict(np.reshape(df,(-1,df.shape[1])))

#Now let us calculate the returns of the day.
df['Return']= np.log(df['close']/df['close'].shift(1))

#Then, create a dataframe called Regimes which will have the OHLC and Return values along with the corresponding regime classification.
Regimes=pd.DataFrame(regime,columns=['Regime'],index=df.index)\
                     .join(df, how='inner')\
                          .assign(market_cu_return=df.Return.cumsum())\
                                 .reset_index(drop=False)\
                                             .rename(columns={'index':'Date'})

#After this, let us create a list called ‘order’ that has the values corresponding to the regime classification, and then plot these values to see how well the algo has classified.
order=[0,1,2,3]
fig = sns.FacetGrid(data=Regimes,hue='Regime',hue_order=order,aspect=2,size= 4)
fig.map(plt.scatter,'Date','market_cu_return', s=4).add_legend()
plt.show()
#The blue zone is the low volatility or the sideways zone
#The red zone is high volatility zone or panic zone.
#The orange zone is a breakout zone.
#The green zone: Not entirely sure.
for i in order:
    print('Mean for regime %i: '%i,unsup.means_[i][0])
    print('Co-Variancefor regime %i: '%i,(unsup.covariances_[i]))

#Regime 0: Low mean and High covariance.
#Regime 1: High mean and High covariance.
#Regime 2: High mean and Low covariance.
#Regime 3: Low mean and Low covariance.
    
#Now finally we have to create our own strategy, before this we have to install Ta-Lib module and Fix-Yahoo module
from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn import mixture as mix
import seaborn as sns 
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import fix_yahoo_finance

df= web.get_data_yahoo('ETFC',start= '2004-01-01', end='2019-05-01')
df=df[['Open','High','Low','Close']]

n = 10
t = 0.8
split =int(t*len(df))

#Next, Trading Strategy: I shifted the High, Low and Close columns by 1, to access only the past data. After this, I created various technical indicators such as, RSI, SMA, ADX, Correlation, Parabolic SAR, and the Return of the past 1- day on an Open to Open basis.

df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)
df['close']=df['Close'].shift(1)
df['RSI']=ta.RSI(np.array(df['close']), timeperiod=n)
df['BBANDSl'],df['BBANDSm'],df['BBANDSh']=ta.BBANDS(np.array(df['close']), timeperiod=n, nbdevup=2, nbdevdn=2)
df['SMA']= df['close'].rolling(window=n).mean()
df['Corr']= df['SMA'].rolling(window=n).corr(df['close'])
df['SAR']=ta.SAR(np.array(df['high']),np.array(df['low']),\
                  0.2,0.2)
df['ADX']=ta.ADX(np.array(df['high']),np.array(df['low']),\
                  np.array(df['close']), timeperiod =n)
df['Corr'][df.Corr>1]=1
df['Corr'][df.Corr<-1]=-1 
df['Return']= np.log(df['Open']/df['Open'].shift(1))

#Next, I printed the data frame
print(df.head())

df=df.dropna()



ss= StandardScaler()
unsup = mix.GaussianMixture(n_components=4, 
                            covariance_type="spherical", 
                            n_init=100, 
                            random_state=42)
df=df.drop(['High','Low','Close'],axis=1)
unsup.fit(np.reshape(ss.fit_transform(df[:split]),(-1,df.shape[1])))
regime = unsup.predict(np.reshape(ss.fit_transform(df[split:]),\
                                                   (-1,df.shape[1])))

Regimes=pd.DataFrame(regime,columns=['Regime'],index=df[split:].index)\
                     .join(df[split:], how='inner')\
                          .assign(market_cu_return=df[split:]\
                                  .Return.cumsum())\
                                  .reset_index(drop=False)\
                                  .rename(columns={'index':'Date'})

order=[0,1,2,3]
fig = sns.FacetGrid(data=Regimes,hue='Regime',hue_order=order,aspect=2,size= 4)
fig.map(plt.scatter,'Date','market_cu_return', s=4).add_legend()
plt.show()

for i in order:
    print('Mean for regime %i: '%i,unsup.means_[i][0])
    print('Co-Variance for regime %i: '%i,(unsup.covariances_[i]))

print(Regimes.head())
#The strategy executes a long position whenever the signal is 1, and a short position whenever the signal is -1.
#Now we will create the cumulative returns for the strategy, and the market.
ss1 =StandardScaler()
columns =Regimes.columns.drop(['Regime','Date'])    
Regimes[columns]= ss1.fit_transform(Regimes[columns])
Regimes['Signal']=0
Regimes.loc[Regimes['Return']>0,'Signal']=1
Regimes.loc[Regimes['Return']<0,'Signal']=-1
Regimes['return'] = Regimes['Return'].shift(1)
Regimes=Regimes.dropna()
       
cls= SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

split2= int(.8*len(Regimes))

X = Regimes.drop(['Signal','Return','market_cu_return','Date'], axis=1)
y= Regimes['Signal']

cls.fit(X[:split2],y[:split2])

p_data=len(X)-split2

df['Pred_Signal']=0
df.iloc[-p_data:,df.columns.get_loc('Pred_Signal')]=cls.predict(X[split2:])

print(df['Pred_Signal'][-p_data:])

df['str_ret'] =df['Pred_Signal']*df['Return'].shift(-1)

df['strategy_cu_return']=0.
df['market_cu_return']=0.
df.iloc[-p_data:,df.columns.get_loc('strategy_cu_return')] \
       = np.nancumsum(df['str_ret'][-p_data:])
df.iloc[-p_data:,df.columns.get_loc('market_cu_return')] \
       = np.nancumsum(df['Return'][-p_data:])
Sharpe = (df['strategy_cu_return'][-1]-df['market_cu_return'][-1])\
           /np.nanstd(df['strategy_cu_return'][-p_data:])

plt.plot(df['strategy_cu_return'][-p_data:],color='g',label='Strategy Returns')
plt.plot(df['market_cu_return'][-p_data:],color='r',label='Market Returns')
plt.figtext(0.14,0.9,s='Sharpe ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()
