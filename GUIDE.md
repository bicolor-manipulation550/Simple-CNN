# Инструкция по использованию обученной модели

Этот гайд поможет вам загрузить уже обученную модель и использовать её для классификации собственных изображений.

## Требования
Для работы вам понадобятся установленные библиотеки:
```bash
pip install tensorflow numpy pillow
```
## Перенесите файл модели и изображения для распознавания в папку проекта

## Загрузка модели в Python

Вам не нужно заново прописывать архитектуру слоев. Просто используйте функцию load_model:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка файла модели
model = load_model('cifar10_model.keras')
print("Модель успешно загружена!")
```
## Подготовка изображения

Модель ожидает картинку строгого формата: 32x32 пикселя и 3 канала (RGB)

```python
import numpy as np
from tensorflow.keras.preprocessing import image

def prepare_image(img_path):

    # Загружаем и меняем размер под вход модели (32x32)
    img = image.load_img(img_path, target_size=(32, 32))

    # Преобразуем в массив и нормализуем (0-1)
    img_array = image.img_to_array(img) / 255.0

    # Добавляем размерность батча (из (32,32,3) делаем (1,32,32,3))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

## Получение предсказания
Список классов CIFAR-10:
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

```python
classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

img = prepare_image('your_image.jpg')
predictions = model.predict(img)
score = tf.nn.softmax(predictions[0]) # Превращаем в вероятности

print(f"Это изображение скорее всего: {classes[np.argmax(score)]}")
print(f"Уверенность: {100 * np.max(score):.2f}%")
```
