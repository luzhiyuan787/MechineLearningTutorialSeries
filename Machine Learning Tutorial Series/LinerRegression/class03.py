#-*-coding:utf-8-*-
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import time

style.use('ggplot')

quandl.ApiConfig.api_key = "x2o2VQTiyq9LmtfBN9Z7"
df = quandl.get('WIKI/AAPL')
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
### 待预测数据集和训练数据集
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
###
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately) ### 根据待预测数据集预测出的数据集
df['Forecast'] = np.nan

last_date = df.iloc[-1].name ## 取出最新一行的时间
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day ## 取出最新一行的下一天时间

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i] ## 创建新的预测数据 并赋预测值


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# line 51
# test_set = [0,1,2,3,4]
#
# df = DataFrame(np.random.randn(5, 4), columns=['A', 'B', 'C', 'D'])
#
# for i in test_set:
#     next_data = 5
#     df.loc[next_data] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
#     print (df)
#     next_data += 1

