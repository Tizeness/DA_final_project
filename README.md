## Итоговый проект: прогнозирование цены Bitcoin

Проект выполнен в рамках программы «Аналитик данных» (НИУ ВШЭ).
Цель работы — построить и сравнить различные модели прогнозирования цены криптовалюты **Bitcoin (BTC-USD)** на основе временных рядов и дополнительных факторов.

---

## Содержание проекта

* **Сбор и очистка данных**

  * Источник: Yahoo Finance (BTC-USD, золото, нефть, INR).
  * Добавлены технические индикаторы (SMA, ATR, MACD и др.).
  * Финальный датасет: `final_merged_correllated.csv`.

* **Исследовательский анализ данных (EDA)**

  * Временные графики (Close, Volume, SMA).
  * Корреляционный анализ (BTC vs Gold, Oil, INR).
  * Проверка сезонности, трендов, волатильности.

* **Моделирование**

  * Статистические модели: **ARIMA, SARIMA, Auto-ARIMA**.
  * Модели машинного обучения: **Linear Regression, Random Forest, XGBoost**.
  * Нейросети: **LSTM (однофакторная и мультифакторная)**.

* **Оценка качества**

  * Метрики: **RMSE, MAE, MAPE**.
  * Сравнение моделей на тестовом промежутке (180 дней).
  * Анализ ошибок (Residuals) по времени и распределению.

* **Визуализация**

  * Факт vs прогнозы по всем моделям.
  * Ошибки (линии и гистограммы).
  * Корреляционные тепловые карты.

---

## Результаты

* Наилучшие результаты на горизонте t+1 показали:

  * **Линейная регрессия** — MAPE ≈ 1,9%.
  * **LSTM (Close)** — MAPE ≈ 3,3%.
* ARIMA/SARIMA/Auto-ARIMA показали ограниченные результаты (слабая сезонность крипторынка).
* Random Forest и XGBoost уступили по точности из-за переобучения.
* Простые модели оказались наиболее надёжными для краткосрочного прогноза.

---

## Технологии

* **Python**: pandas, numpy, matplotlib, seaborn
* **ML/TS**: statsmodels (ARIMA, SARIMA), pmdarima (Auto-ARIMA), scikit-learn (LR, RF, XGBoost), keras/tensorflow (LSTM)
* **Инструменты**: Jupyter Notebook, Yandex DataLens (дашборд)

---

## Заключение

Проект показал, что:

* Методы анализа временных рядов и машинного обучения применимы для крипторынка.
* Простые модели часто работают лучше сложных на коротких горизонтах.
* Перспективные направления: гибридные модели и расширение набора признаков.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Загружаем данные
df = pd.read_csv("data/final_merged_correllated.csv", parse_dates=["Date"]).sort_values("Date")
values = df["Close"].astype(float).values.reshape(-1,1)

# 2. Масштабируем
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 3. Формируем выборку с окном 30 дней
def create_dataset(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 0])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled, 30)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Делим на train/test (последние 180 дней)
TEST_DAYS = 180
X_train, X_test = X[:-TEST_DAYS], X[-TEST_DAYS:]
y_train, y_test = y[:-TEST_DAYS], y[-TEST_DAYS:]

# 5. Модель LSTM
model = Sequential([
    LSTM(50, activation="tanh", input_shape=(30, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# 6. Прогноз
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# 7. Метрики
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
mape = (np.abs((y_test_inv - y_pred_inv) / y_test_inv)).mean() * 100

print(f"LSTM → RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

