import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import kagglehub

# --------- Загрузка датасета ---------
print("Загрузка датасета...")
path = kagglehub.dataset_download("zeesolver/cloiud-dataset")
print("Путь к датасету:", path)

# --------- Исследуем структуру датасета ---------
print("\nИсследуем структуру датасета...")
for root, dirs, files in os.walk(path, topdown=True):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if len(files) > 0:
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print(f'{indent}  изображений: {len(image_files)}')


# --------- Загрузка изображений ---------
def load_images(base_path, img_size=(64, 64)):
    images = []
    labels = []

    # Словарь классов
    CLASS_MAP = {
        0: ["clear", "sunny", "blue"],
        1: ["cloud", "cloudy", "overcast", "sky"]
    }

    def detect_class(name):
        name = name.lower()
        for class_idx, keys in CLASS_MAP.items():
            if any(k in name for k in keys):
                return class_idx
        return None

    for root, dirs, files in os.walk(base_path):
        folder_name = os.path.basename(root)
        folder_class = detect_class(folder_name)

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_idx = folder_class
                if class_idx is None:
                    class_idx = detect_class(filename)

                if class_idx is not None:
                    try:
                        img_path = os.path.join(root, filename)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(img_size)
                        images.append(np.array(img))
                        labels.append(class_idx)
                    except:
                        continue

    return np.array(images), np.array(labels)


# Загружаем данные
train_data, train_labels = load_images(path)
print(f"\nЗагружено изображений: {len(train_data)}")

if len(train_data) == 0:
    print("ОШИБКА: Не удалось загрузить изображения! Проверьте структуру датасета.")
    exit()

# --------- Предобработка ---------
train_data = train_data.astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes=2)

# --------- Модель ---------
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --------- Обучение ---------
print("\nНачинаем обучение...")
model.fit(train_data, train_labels, epochs=20, batch_size=2, validation_split=0.2)

# --------- Сохранение ---------
model.save('cnn_cloud_model.h5')
print("\nМодель сохранена как 'cnn_cloud_model.h5'.")
