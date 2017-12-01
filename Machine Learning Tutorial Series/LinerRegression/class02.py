#-*-coding:utf-8-*-
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# 训练和测试 last_unix = time.mktime(last_date.timetuple())

quandl.ApiConfig.api_key = "x2o2VQTiyq9LmtfBN9Z7"
df = quandl.get('WIKI/AAPL')
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']] # 选择列的一个子集

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0 # 基于收盘价的百分比极差，这是我们对于波动的粗糙度量

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # 每日百分比变化

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # 新的数据集

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # 将数据中为空的数据置为—99999
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out) # 整个长度的 1% 的数据上移后作为预测值

df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) # 特征

y = np.array(df['label']) # 特征的标签

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# 特征的训练集、测试集 标签的训练集和测试集
# 整个长度的 1% 的数据上移后作为标签  表示当前情况对以后的市场的影响

clf = LinearRegression(n_jobs=-1) # 分类器

clf.fit(X_train, y_train) # 拟合了训练特征和训练标签

confidence = clf.score(X_test, y_test) # 准确率

print(confidence)

