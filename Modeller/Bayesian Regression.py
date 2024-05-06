import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Veri setini yükleme ve ön işleme
veri = pd.read_csv(r"C:\Users\ASUS\Desktop\AI CLUB DATATHON\CSV\ail_frx.csv")
veri['open'] = veri['open'].str.replace('"', '').astype(float)
veri['high'] = veri['high'].str.replace('"', '').astype(float)
veri['low'] = veri['low'].str.replace('"', '').astype(float)
veri['close'] = veri['close'].str.replace('"', '').astype(float)

# open sütunundaki NaN değerlerini ortalama ile doldurma
veri['open'].fillna(veri['open'].mean(), inplace=True)
veri['high'].fillna(veri['high'].mean(), inplace=True)
veri['low'].fillna(veri['low'].mean(), inplace=True)
veri['close'].fillna(veri['close'].mean(), inplace=True)

X = veri[['open', 'low', 'high']]
y = veri['close']

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modeli oluşturma ve eğitme
model = BayesianRidge()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Bayesian Regression RMSE:", rmse)
