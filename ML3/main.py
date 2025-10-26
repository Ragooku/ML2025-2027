from ucimlrepo import fetch_ucirepo
import random
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# fetch dataset
concrete_compressive_strength = fetch_ucirepo(id=165)

# data (as pandas dataframes)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# metadata
print(concrete_compressive_strength.metadata)

# variable information
print(concrete_compressive_strength.variables)


def extract_float_value(value):
    if isinstance(value, str):
        if ':' in value:
            try:
                return float(value.split(':')[-1])
            except (ValueError, IndexError):
                return np.nan
        elif ';' in value:
            try:
                return float(value.split(';')[-1])
            except (ValueError, IndexError):
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan


def load_and_prepare_data():
    # Используем глобальные переменные X и y
    global X, y

    X_df = X.copy()
    y_df = y.copy()

    X_df = X_df.map(extract_float_value)
    # Заполнение NaN: для каждого столбца заполняем медианой, если столбец не полностью NaN, иначе 0.
    X_df = X_df.apply(lambda col: col.fillna(col.median() if not col.isnull().all() else 0))
    X_df = X_df.astype(float)

    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
        y_df = y_df.iloc[:, 0]
    elif isinstance(y_df, pd.DataFrame) and y_df.shape[1] > 1:
        raise ValueError("y_df содержит несколько столбцов. Ожидается одномерный target.")

    y_df = y_df.map(extract_float_value)
    y_df = y_df.fillna(y_df.median() if not y_df.isnull().all() else 0)
    y_df = y_df.astype(float)

    return X_df, y_df


X, y = load_and_prepare_data()

# 1) Разделение на тестовую и обучающие выборки

# Общее количество образцов
n_samples = X.shape[0]

# Массив индексов от 0 до n_samples-1
indices = np.arange(n_samples)

# Перемешиваем индексы случайным образом
np.random.shuffle(indices)

# Применяем перемешанные индексы к X и y (DataFrame/Series)
X_shuffled = X.iloc[indices]
y_shuffled = y.iloc[indices]

# Разделяем перемешанные данные
train_size = int(n_samples * 0.8)

X_train = X_shuffled[:train_size]
X_test = X_shuffled[train_size:]

y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

# 2) Обучение модели по линейной регрессии

regressor = LinearRegression().fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

print(f"Coefficient of determination TRAIN: {r2_score(y_train, y_train_pred):.2f}")
print(f"Coefficient of determination TEST: {r2_score(y_test, y_test_pred):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="black", alpha=0.6, label="Фактические vs. Прогноз")

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle='--', linewidth=2,
         label="Идеальный прогноз (y=x)")

plt.xlabel("Истинные значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("Истинные значения vs. Предсказанные значения")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

degrees = range(1, 3)
r2_train_list = []
r2_test_list = []

for degree in degrees:
    print(degree)
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    r2_train_list.append(r2_score(y_train, y_train_pred))
    r2_test_list.append(r2_score(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.plot(degrees, r2_train_list, marker='o', label="Train R2")
plt.plot(degrees, r2_test_list, marker='o', label="Test R2")
plt.xlabel("Degree of Polynomial Features")
plt.ylabel("R^2")
plt.legend()
plt.grid(True)
plt.show()

# 4) Построение модели с использованием полиноминальной функции
# 5) Построение модели с использованием регуляризации
degree = 2
alphas = np.logspace(-4, 3, 10)  # диапазон коэффициентов регуляризации

r2_train_list = []
r2_test_list = []

for alpha in alphas:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, max_iter=10000))
    ])

    # Обучаем модель на train
    pipeline.fit(X_train, y_train)

    # Предсказания
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # R^2
    r2_train_list.append(r2_score(y_train, y_train_pred))
    r2_test_list.append(r2_score(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, r2_train_list, marker='o', label="Train R^2")
plt.semilogx(alphas, r2_test_list, marker='o', label="Test R^2")
plt.xlabel("Alpha (коэффициент регуляризации)")
plt.ylabel("R^2")
plt.title(f"Ridge Regression (Polynomial degree={degree})")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

best_index = np.argmax(r2_test_list)
best_alpha = alphas[best_index]

print(f"Наилучший alpha: {best_alpha:.4f}")
