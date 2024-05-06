import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Veri setini yükleme ve ön işleme
veri = pd.read_csv(r"C:\Users\ASUS\Desktop\AI CLUB DATATHON\CSV\ail_frx.csv")

# NaN değerleri ortalama ile doldurma
veri['open'] = veri['open'].str.replace('"', '').astype(float)
veri['high'] = veri['high'].str.replace('"', '').astype(float)
veri['low'] = veri['low'].str.replace('"', '').astype(float)
veri['close'] = veri['close'].str.replace('"', '').astype(float)

veri['open'].fillna(veri['open'].mean(), inplace=True)
veri['high'].fillna(veri['high'].mean(), inplace=True)
veri['low'].fillna(veri['low'].mean(), inplace=True)
veri['close'].fillna(veri['close'].mean(), inplace=True)

# Giriş ve çıkış verilerini ayırma
X = veri[['open', 'low', 'high']]
y = veri['close']

# Eğitim ve test setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Özellik ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelin oluşturulması
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test_scaled)

# Tahminlerin performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)


# RMSE: 0.6424700421979755