import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('cnn_cloud_model.h5')

def preprocess_image(image_path, img_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    print("=" * 50)
    print("ПРОВЕРКА МОДЕЛИ: ЧИСТОЕ НЕБО / ОБЛАЧНОЕ НЕБО")
    print("=" * 50)

    while True:
        print("\n1. Проверить изображение")
        print("2. Выход")
        choice = input("Введите номер: ")

        if choice == '1':
            image_path = input("Введите путь к изображению: ")

            try:
                img_array = preprocess_image(image_path)
                pred = model.predict(img_array, verbose=0)[0]

                clear_prob = pred[0]
                cloudy_prob = pred[1]

                print("\nРезультат:")
                print(f"Чистое небо:   {clear_prob:.4f}")
                print(f"Облачное небо: {cloudy_prob:.4f}")

                if clear_prob > cloudy_prob:
                    print("Предсказано: ЧИСТОЕ НЕБО")
                else:
                    print("Предсказано: ОБЛАЧНОЕ НЕБО")

            except Exception as e:
                print(f"Ошибка: {e}")
                print("Проверь путь к файлу.")

        elif choice == '2':
            break

        else:
            print("Некорректный ввод.")

if __name__ == "__main__":
    main()
