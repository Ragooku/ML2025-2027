import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


# Функция для загрузки данных из папки dataset
def load_dataset_from_folder(folder_path):
    """
    Загружает все .data файлы из указанной папки и объединяет их в один датасет
    """
    all_data = []

    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        if filename.endswith('.data'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Читаем файл, предполагая, что данные разделены запятыми
                data = pd.read_csv(file_path, header=None)
                all_data.append(data)
                print(f"Загружен файл: {filename} с {len(data)} примерами")
            except Exception as e:
                print(f"Ошибка при загрузке {filename}: {e}")

    if not all_data:
        raise ValueError("Не найдено .data файлов в указанной папке")

    # Объединяем все данные
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Всего загружено примеров: {len(combined_data)}")

    return combined_data


# Функция для оценки кластеризации с расширенными метриками
def evaluate_clustering(model, data, true_labels=None):
    labels = model.fit_predict(data)

    # Проверка, что есть более 1 кластера
    if len(set(labels)) > 1 and -1 in labels:
        core_labels = labels[labels != -1]
        core_data = data[labels != -1]
        silhouette = silhouette_score(core_data, core_labels)
        calinski = calinski_harabasz_score(core_data, core_labels)
        davies = davies_bouldin_score(core_data, core_labels)
    elif len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
    else:
        silhouette = -1
        calinski = -1
        davies = float('inf')

    # Метрики, требующие истинные метки
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
    else:
        ari = -1
        nmi = -1

    return labels, silhouette, calinski, davies, ari, nmi


# Загрузка данных из папки dataset
dataset_folder = 'dataset'  # Путь к папке с данными
data = load_dataset_from_folder(dataset_folder)

# Предполагаем, что последний столбец - это метка класса
X = data.iloc[:, :-1]  # Все столбцы кроме последнего
y = data.iloc[:, -1]  # Последний столбец - метка

print(f"Размерность данных: {X.shape}")
print(f"Уникальные классы: {np.unique(y)}")

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Преобразуем в плотный массив для совместимости
X_scaled_dense = X_scaled

# Визуализация данных после PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_dense)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=30)
plt.title("Визуализация исходных данных (PCA)")
plt.colorbar(label='Класс')
plt.show()

# 1. KMeans
print("\n=== KMeans кластеризация ===")
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []
metrics_kmeans = []

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(kmeans, X_scaled_dense, y)
    print(f'KMeans с k={k}:')
    print(f'  Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}, Davies-Bouldin: {davies:.3f}')
    print(f'  ARI: {ari:.3f}, NMI: {nmi:.3f}')

    labels_for_k.append(labels)
    metrics_kmeans.append({'silhouette': silhouette, 'calinski': calinski, 'davies': davies, 'ari': ari, 'nmi': nmi})

    if silhouette > best_score_kmeans:
        best_score_kmeans = silhouette
        best_kmeans = kmeans
        best_labels_kmeans = labels
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с Silhouette: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(15, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'KMeans k={k}\nSil: {metrics_kmeans[i]["silhouette"]:.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# 2. Agglomerative Clustering
print("\n=== Agglomerative кластеризация ===")
agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None
labels_list = []
metrics_agg = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(agg, X_scaled_dense, y)
    print(f'Agglomerative с n_clusters={n}:')
    print(f'  Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}, Davies-Bouldin: {davies:.3f}')
    print(f'  ARI: {ari:.3f}, NMI: {nmi:.3f}')

    labels_list.append(labels)
    metrics_agg.append({'silhouette': silhouette, 'calinski': calinski, 'davies': davies, 'ari': ari, 'nmi': nmi})

    if silhouette > best_score_agg:
        best_score_agg = silhouette
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с Silhouette: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(15, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'Agglomerative n={n}\nSil: {metrics_agg[i]["silhouette"]:.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# 3. SpectralClustering
print("\n=== Spectral Clustering ===")
spectral_params = [2, 3, 4, 5, 6]
best_score_spectral = -1
best_labels_spectral = []
best_n_spectral = None
labels_list_spectral = []
metrics_spectral = []

for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(spectral, X_scaled_dense, y)
    print(f'SpectralClustering с n_clusters={n}:')
    print(f'  Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}, Davies-Bouldin: {davies:.3f}')
    print(f'  ARI: {ari:.3f}, NMI: {nmi:.3f}')

    labels_list_spectral.append(labels)
    metrics_spectral.append({'silhouette': silhouette, 'calinski': calinski, 'davies': davies, 'ari': ari, 'nmi': nmi})

    if silhouette > best_score_spectral:
        best_score_spectral = silhouette
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с Silhouette: {best_score_spectral:.3f}')

# Визуализация результатов для всех вариантов
fig, axes = plt.subplots(1, len(spectral_params), figsize=(15, 4))
for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'Spectral n={n}\nSil: {metrics_spectral[i]["silhouette"]:.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# Визуализация лучших кластеризаций
plt.figure(figsize=(15, 5))

# Находим лучшие метрики для каждого метода
best_metrics_kmeans = next(m for i, m in enumerate(metrics_kmeans) if kmeans_params[i] == best_k)
best_metrics_agg = next(m for i, m in enumerate(metrics_agg) if agg_params[i] == best_n_agg)
best_metrics_spectral = next(m for i, m in enumerate(metrics_spectral) if spectral_params[i] == best_n_spectral)

# KMeans
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis')
plt.title(f'KMeans (k={best_k})\nSil: {best_score_kmeans:.3f}, ARI: {best_metrics_kmeans["ari"]:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

# Agglomerative Clustering
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='viridis')
plt.title(f'Agglomerative (n={best_n_agg})\nSil: {best_score_agg:.3f}, ARI: {best_metrics_agg["ari"]:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

# SpectralClustering
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_spectral, cmap='viridis')
plt.title(f'Spectral (n={best_n_spectral})\nSil: {best_score_spectral:.3f}, ARI: {best_metrics_spectral["ari"]:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

plt.tight_layout()
plt.show()

# Итог: определение лучшего метода
scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'SpectralClustering': best_score_spectral
}

best_method = max(scores, key=scores.get)
print(f'\n=== РЕЗУЛЬТАТЫ ===')
print(f'Лучший метод кластеризации: {best_method} с Silhouette: {scores[best_method]:.3f}')
print(f"KMeans: Silhouette: {scores['KMeans']:.3f}, ARI: {best_metrics_kmeans['ari']:.3f}")
print(f"Agglomerative: Silhouette: {scores['Agglomerative']:.3f}, ARI: {best_metrics_agg['ari']:.3f}")
print(f"Spectral: Silhouette: {scores['SpectralClustering']:.3f}, ARI: {best_metrics_spectral['ari']:.3f}")

# Сводная таблица метрик
print(f"\n=== СВОДКА МЕТРИК ===")
methods = ['KMeans', 'Agglomerative', 'SpectralClustering']
best_metrics = [best_metrics_kmeans, best_metrics_agg, best_metrics_spectral]

print(f"{'Метод':<15} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12} {'ARI':<12} {'NMI':<12}")
print("-" * 75)
for method, metrics in zip(methods, best_metrics):
    print(f"{method:<15} {metrics['silhouette']:<12.3f} {metrics['calinski']:<12.1f} "
          f"{metrics['davies']:<12.3f} {metrics['ari']:<12.3f} {metrics['nmi']:<12.3f}")

# ВИЗУАЛИЗАЦИЯ СВОДКИ МЕТРИК ДЛЯ ЛУЧШИХ ПАРАМЕТРОВ
print(f"\n=== ВИЗУАЛИЗАЦИЯ СВОДКИ МЕТРИК ===")

# Создаем график со сводной таблицей
plt.figure(figsize=(14, 8))
ax = plt.subplot(111)
ax.axis('tight')
ax.axis('off')

# Подготавливаем данные для таблицы
table_data = []
for method, metrics in zip(methods, best_metrics):
    table_data.append([
        method,
        f"{metrics['silhouette']:.3f}",
        f"{metrics['calinski']:.1f}",
        f"{metrics['davies']:.3f}",
        f"{metrics['ari']:.3f}",
        f"{metrics['nmi']:.3f}"
    ])

# Определяем количество столбцов
num_cols = len(table_data[0])  # Должно быть 6
num_rows = len(table_data) + 1  # +1 для заголовков

print(f"Количество столбцов: {num_cols}")
print(f"Количество строк: {num_rows}")

# Создаем таблицу
table = ax.table(cellText=table_data,
                colLabels=['Метод', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'ARI', 'NMI'],
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.1, 0.8, 0.8])  # Уменьшаем bbox для безопасности

# Настраиваем внешний вид таблицы
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Выделяем заголовки (правильно индексируем ячейки)
for i in range(num_cols):
    # Заголовки находятся в первой строке (индекс 0)
    table[(0, i)].set_facecolor('#4C72B0')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Выделяем лучший метод
best_method_index = methods.index(best_method)
for i in range(num_cols):
    # Данные начинаются со строки 1 (индекс 1)
    table[(best_method_index + 1, i)].set_facecolor('#e6f3ff')
    table[(best_method_index + 1, i)].set_text_props(weight='bold')

plt.title('СВОДКА МЕТРИК ДЛЯ ЛУЧШИХ ПАРАМЕТРОВ\n', size=16, weight='bold', pad=30)
plt.tight_layout()
plt.show()

# Дополнительная информация о данных
print(f"\n=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Общее количество примеров: {len(X)}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Уникальные классы в данных: {np.unique(y)}")
print(f"Распределение классов:")
print(pd.Series(y).value_counts().sort_index())