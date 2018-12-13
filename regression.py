import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df["HL_PCT"] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', "HL_PCT", "PCT_change", "Adj. Volume"]] #görmek istediğimiz etiketleri belirttik, yazdıralım.
df.fillna(-9999, inplace=True)
print(df.head())

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df['label'].head())

x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
x = preprocessing.scale(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, train_size=0.2)
classification = LinearRegression()
classification.fit(x_train,y_train)
accuracy = classification.score(x_test, y_test)
print(accuracy)
df['Adj. Close'].plot
df['label'].plot