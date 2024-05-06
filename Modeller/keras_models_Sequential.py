import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Veri setini yükleme ve ön işleme
veri = pd.read_csv(r"C:/Users/ASUS/Desktop/AI CLUB DATATHON/CSV/zzc_ail.csv")

veri['open'] = veri['open'].str.replace('"', '').astype(float)
veri['high'] = veri['high'].str.replace('"', '').astype(float)
veri['low'] = veri['low'].str.replace('"', '').astype(float)
veri['close'] = veri['close'].str.replace('"', '').astype(float)

# Gerekli sütunları seçme
veri = veri[['open', 'low', 'high', 'close']]

# NaN değerleri ortalama ile doldurma
veri.fillna(veri.mean(), inplace=True)

# Bağımsız ve bağımlı değişkenleri tanımlama
X = veri[['open', 'low', 'high']].values
y = veri['close'].values

# Verileri normalize etme
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1,1)).reshape(-1)

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Modelin oluşturulması
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


# Modelin performansını değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)




# veri = pd.read_csv(r"C:\Users\ASUS\Desktop\AI CLUB DATATHON\CSV\ail_frx.csv")
# RMSE: 0.0784939565510259


# veri = pd.read_csv(r"C:/Users/ASUS/Desktop/AI CLUB DATATHON/CSV/bk_frx.csv")
# RMSE: 0.014220362411334137

# veri = pd.read_csv(r"C:\Users\ASUS\Desktop\AI CLUB DATATHON\CSV\crp_ail.csv")
# RMSE: 0.05298931673090933


# veri = pd.read_csv(r"C:/Users/ASUS/Desktop/AI CLUB DATATHON/CSV/zzc_ail.csv")
# RMSE: 0.0487921137408981