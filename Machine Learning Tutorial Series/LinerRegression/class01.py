#-*-coding:utf-8-*-
import quandl

# regression 回归
# 引言和数据
quandl.ApiConfig.api_key = "x2o2VQTiyq9LmtfBN9Z7"
df = quandl.get('WIKI/AAPL')
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']] # 选择列的一个子集

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0 # 基于收盘价的百分比极差，这是我们对于波动的粗糙度量

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # 每日百分比变化

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # 新的数据集

print(df.head())