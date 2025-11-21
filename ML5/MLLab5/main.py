import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
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


# Функция для оценки кластеризации
def evaluate_clustering(model, data):
    labels = model.fit_predict(data)
    # Проверка, что есть более 1 кластера
    if len(set(labels)) > 1 and -1 in labels:
        core_labels = labels[labels != -1]
        core_data = data[labels != -1]
        score = silhouette_score(core_data, core_labels)
    elif len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1
    return labels, score


# 1. KMeans
print("\n=== KMeans кластеризация ===")
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []  # список для хранения меток для каждого k

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels, score = evaluate_clustering(kmeans, X_scaled_dense)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {score:.3f}')
    labels_for_k.append(labels)  # сохраняем метки для каждого k
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_kmeans = kmeans
        best_labels_kmeans = labels  # метки для лучшего k
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(15, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'KMeans k={k}')
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

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, score = evaluate_clustering(agg, X_scaled_dense)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list.append(labels)
    if score > best_score_agg:
        best_score_agg = score
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(15, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'Agglomerative n={n}')
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

for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
    labels, score = evaluate_clustering(spectral, X_scaled_dense)
    print(f'SpectralClustering с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list_spectral.append(labels)
    if score > best_score_spectral:
        best_score_spectral = score
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с коэффициентом: {best_score_spectral:.3f}')

# Визуализация результатов для всех вариантов
fig, axes = plt.subplots(1, len(spectral_params), figsize=(15, 4))
for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'Spectral n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# Визуализация лучших кластеризаций
plt.figure(figsize=(15, 5))

# KMeans
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis')
plt.title(f'KMeans (k={best_k})\nSilhouette: {best_score_kmeans:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

# Agglomerative Clustering
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='viridis')
plt.title(f'Agglomerative (n={best_n_agg})\nSilhouette: {best_score_agg:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

# SpectralClustering
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_spectral, cmap='viridis')
plt.title(f'Spectral (n={best_n_spectral})\nSilhouette: {best_score_spectral:.3f}')
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
print(f'Лучший метод кластеризации: {best_method} с коэффициентом {scores[best_method]:.3f}')
print(f"KMeans: {scores['KMeans']:.3f}")
print(f"Agglomerative: {scores['Agglomerative']:.3f}")
print(f"Spectral: {scores['SpectralClustering']:.3f}")

# Дополнительная информация о данных
print(f"\n=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Общее количество примеров: {len(X)}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Уникальные классы в данных: {np.unique(y)}")
print(f"Распределение классов:")
print(pd.Series(y).value_counts().sort_index())