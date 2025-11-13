# Лабораторная работа 4. Основы нейронных сетей
## Задание
Перед выполнением лабораторной работы необходимо загрузить набор данных в соответствии с вариантом на диск

1. Написать программу, которая разделяет исходную выборку на обучающую и тестовую (training set, validation set, test set), если такое разделение не предусмотрено предложенным набором данных.
2. Произвести масштабирование признаков (scaling).
3. С использованием библиотеки scikit-learn обучить 2 модели нейронной сети (Perceptron и MLPClassifier) по обучающей выборке. Перед обучением необходимо осуществить масштабирование признаков. Пример MLPClassifier Пример и описание Perceptron
4. Проверить точность модели по тестовой выборке.
5. Провести эксперименты и определить наилучшие параметры коэффициента обучения, параметра регуляризации, функции оптимизации. Данные экспериментов необходимо представить в отчете (графики, ход проведения эксперимента, выводы).

## Вариант 7
Banknote authentication

## Загрузка dataset
```
from ucimlrepo import fetch_ucirepo

banknote_authentication = fetch_ucirepo(id=267)

# Данные
X = banknote_authentication.data.features
y = banknote_authentication.data.targets
```
x - Features, Признаки. 

y - Targets, Цели
## Разделение исходной выборки на обучающую и тестовую
В документации sklearn под тестовую выборку выделяется 20% данных. В программе используем стандартную функцию train_test_split.
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
## Масштабирование признаков

Масштабирование признаков — это процесс приведения всех числовых признаков к одинаковому масштабу. В данном коде используется стандартизация - один из самых популярных методов масштабирования.

Стандартизация данных (приведение к нулевому среднему и единичному стандартному отклонению) (метод масштабирования).

scaler = StandardScaler() — Создание объекта StandardScaler

X_train_scaled = scaler.fit_transform(X_train) — Обучение scaler на тренировочных данных и преобразование тренировочных данных

X_test_scaled = scaler.transform(X_test) — Преобразование тестовых данных с использованием параметров, полученных на тренировочных данных

Масштабирование устраняет смещение из‑за разных единиц измерения, ускоряет обучение и повышает качество моделей, чувствительных к масштабу признаков.

Далее производится стандартизация данных для обеих выборок. Для тренировочной выборки вычисляются среднее значение и стандартное отклонение.
После чего такие же параметры будут применены к тестовой выборке.
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
## Обучение Perceptron и MLPClassifier

Perceptron (Однослойный перцептрон) — модель, имеющая только входной и выходной слои, без скрытых слоев

MLPClassifier (Многослойный перцептрон) — модель, имеющая только входной, выходной и скрытые слои

Отличия:

1. Perceptron: один слой, линейная классификация
2. MLP: несколько слоев, универсальный аппроксиматор (может решать нелинейные задачи)

От однослойных линейных моделей к многослойным нелинейным сетям с обучением через градиенты.

```
# 3) Обучение Perceptron и MLPClassifier
# =========
#Perceptron
print("Обучение Perceptron...")
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

# MLPClassifier с базовыми параметрами
print("Обучение MLPClassifier...")
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

```
## Проверка точности моделей
```
print(f"Точность Perceptron: {accuracy_perceptron:.4f}")
print(f"Точность MLPClassifier: {accuracy_mlp:.4f}")
```

![alt text](MLLab4\1.png)

у датасета хорошие разделенные данные, поэтому он стремится к 1

## Гиперпараметры для эксперимента
hidden_layer_sizes - размеры скрытых слоев нейронной сети

activation - функция активации (relu — функция Rectified Linear Unit, tanh — гиперболический тангенс)

solver — алгоритм оптимизации (adam — адаптивный градиентный спуск, sgd — стохастический градиентный спуск)

alpha — коэффициент регуляризации

learning_rate_init — начальная скорость обучения

sigmoid - преобразует входное значение в диапазон от 0 до 1. σ(x) = 1 / (1 + e^(-x))

softmax - преобразует вектор необработанных оценок в распределение вероятностей по нескольким классам. Функция помогает определить вероятности для каждого класса. Softmax преобразует результаты слоя в выходное распределение вероятностей.

ReLU - Выводит входные данные напрямую, если они положительные, и ноль в противном случае. В скрытых слоях нейронной сети помогает внести нелинейность, позволяющую модели изучать сложные закономерности в данных.

tanh или гиперболический тангенс - нелинейная функция, которая отображает входные данные в диапазон от -1 до 1.

leaky ReLU - вариант ReLU, который допускает небольшой, ненулевой градиент, когда входное значение отрицательно.

Гиперпараметры нейронной сети — это настраиваемые параметры, которые задаются до начала обучения и управляют самим процессом обучения и структурой модели. В работе мы изменяем:

Архитектуру сети (количество и размер скрытых слоев)

Тип функции активации (Rectified Linear Unit = f(x) = max(0, x); Hyperbolic Tangent = f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)); Logistic = f(x) = 1 / (1 + e^(-x)))

Алгоритм оптимизации весов модели (Adaptive Moment Estimation, который имеет адаптивную скорость обучения для каждого параметра; Stochastic Gradient Descent, который Обновляет веса после каждого батча)

Коэффициент регуляризации

Начальную скорость обучения
```
# =========
# 5) Подбор гиперпараметров
# =========
print("Запуск GridSearchCV для подбора гиперпараметров...")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
                           param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train_scaled, y_train)

print("\nЛучшие параметры:", grid_search.best_params_)
print("Лучшая точность на валидационной выборке: {:.4f}".format(grid_search.best_score_))

# Оценка лучшей модели на тестовых данных
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Точность лучшей модели на тестовых данных: {accuracy_best:.4f}")
```

## Визуализация

```
results = pd.DataFrame(grid_search.cv_results_)

# Создание фигуры с графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Результаты экспериментов
ax1.plot(results['mean_test_score'], marker='o', alpha=0.7)
ax1.set_xlabel('Номер эксперимента')
ax1.set_ylabel('Средняя точность')
ax1.set_title('Результаты перебора гиперпараметров')
ax1.grid(True, alpha=0.3)

# 2. Сравнение моделей
models = ['Perceptron', 'MLP (базовый)', 'MLP (лучший)']
accuracies = [accuracy_perceptron, accuracy_mlp, accuracy_best]
bars = ax2.bar(models, accuracies, color=['lightblue', 'lightcoral', 'lightgreen'])
ax2.set_ylabel('Точность')
ax2.set_title('Сравнение точности моделей')
ax2.set_ylim(0, 1)
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Сравнение с базовыми моделями
print("\n" + "="*100)
print("ИТОГОВОЕ СРАВНЕНИЕ:")
print(f"Perceptron: {accuracy_perceptron:.4f}")
print(f"MLPClassifier (базовый): {accuracy_mlp:.4f}")
print(f"MLPClassifier (оптимизированный): {accuracy_best:.4f}")
```
Путем перебора различных комбинаций получаем следующие наилучшие значения:
![alt text](MLLab4/5.png)

![alt text](MLLab4/2.png)

![alt text](MLLab4/3.png)





