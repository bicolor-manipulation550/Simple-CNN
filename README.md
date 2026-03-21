<div align="center">

# Simple-CNN

![GitHub License](https://img.shields.io/github/license/Vanya737/Simple-CNN)
![GitHub Release](https://img.shields.io/github/v/release/Vanya737/Simple-CNN)
![GitHub top language](https://img.shields.io/github/languages/top/Vanya737/Simple-CNN)
![GitHub last commit](https://img.shields.io/github/last-commit/Vanya737/Simple-CNN)
![GitHub Repo stars](https://img.shields.io/github/stars/Vanya737/Simple-CNN)

</div>

## CIFAR-10

![Static Badge](https://img.shields.io/badge/Information--data-gray)
![Static Badge](https://img.shields.io/badge/CIFAR--10-white)
![Static Badge](https://img.shields.io/badge/CNN-orange)

Датасет CIFAR-10. Это один из самых популярных наборов данных для обучения нейросетей начального уровня.  

●	**Количество изображений:** 60 000 штук.  
●	**Размер картинок:** 32x32 пикселя (чуть меньше вашего лимита в 50x50, что делает обучение очень быстрым).  
●	**Цвет:** Полноцветные (RGB).  
●	**Разделение:** 50 000 картинок для обучения и 10 000 для проверки (теста)  

Датасет разбит на **10 строго определенных классов**. Каждому числу (метке) соответствует свой объект:  
  
| Индекс | Класс | Описание |
| :---: | :--- | :--- |
| 0 | Самолет | Пассажирские лайнеры, истребители |
| 1 | Автомобиль | Легковые машины, седаны |
| 2 | Птица | Певчие птицы, водоплавающие |
| 3 | Кот | Домашние кошки (разных окрасов) |
| 4 | Олень | Лесные олени |
| 5 | Собака | Разные породы собак |
| 6 | Лягушка | Обычные лягушки, жабы |
| 7 | Лошадь | Кони, пони |
| 8 | Корабль | Лодки, крейсеры, танкеры |
| 9 | Грузовик | Большие машины, фуры |

**Встроен в библиотеку:** Вам не нужно скачивать файлы вручную — команда datasets.cifar10.load_data() сама загрузит всё необходимое в кэш Python.  

Загрузка датасета **CIFAR-10**  осуществляется через библиотеку **TensorFlow** (а точнее, через её встроенный модуль Keras).

**Сложность:** В отличие от совсем простого MNIST (черно-белые цифры), здесь реальные объекты. Нейросети нужно научиться понимать формы ушей кота или колеса машины, несмотря на разный фон.

```python
from tensorflow.keras import datasets

# Загрузка происходит одной командой
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```

<img src="https://github.com/user-attachments/assets/3dafc0a4-2ad6-4d19-b5f6-211d1feb8bd0" width="500">

## Архитектура CNN

![Static Badge](https://img.shields.io/badge/Python-blue)
![Static Badge](https://img.shields.io/badge/TensorFlow-orange)
![Static Badge](https://img.shields.io/badge/Keras-red)

| № | Слой (Тип) | Конфигурация / Параметры | Выходная форма | Назначение слоя |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Conv2D** | 32 фильтра (3x3), ReLU | (None, 30, 30, 32) | Извлечение базовых признаков (линии, углы) |
| 2 | **MaxPooling2D** | Размер окна (2x2) | (None, 15, 15, 32) | Сжатие данных в 2 раза, выделение главных признаков |
| 3 | **Conv2D** | 64 фильтра (3x3), ReLU | (None, 13, 13, 64) | Поиск более сложных паттернов (формы, текстуры) |
| 4 | **MaxPooling2D** | Размер окна (2x2) | (None, 6, 6, 64) | Вторичное сжатие признаков |
| 5 | **Flatten** | Выравнивание в вектор | (None, 2304) | Преобразование 2D-карт признаков в 1D-массив |
| 6 | **Dense** | 64 нейрона, ReLU | (None, 64) | Полносвязная интерпретация найденных признаков |
| 7 | **Dense (Output)** | 10 нейронов, Softmax | (None, 10) | Финальные оценки для каждого из 10 классов |

Все функции осуществляются через tensorflow
и numpy

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 
```
  
## Обучение

```python
model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))
```
Минимальный порог эпохи: 3 (число полных циклов прохода по датасету)  
Оптимальным будет количество эпох в интервале от 5 до 8
  
## Оптимизация

<div align="center">

| Кол-во эпох | Время обучения, сек. | Примерная точность модели | График |
| :---: | :---: | :--- | :---: |
| 3 | 248,7 | Недообучение | <img src="https://github.com/user-attachments/assets/aac66d6c-3893-41c1-994c-24cfd732bd73" width="180"> |
| 10 | 743,9 | Переобучение | <img src="https://github.com/user-attachments/assets/88eb53f1-61db-4040-9866-9dab7421bf9f" width="180"> |
| 6 | 470,2 | **Оптимальное** | <img src="https://github.com/user-attachments/assets/31b871b7-6bdc-43c3-94e9-9cf7778379f5" width="180"> |
| 5 | 358,2 | Недообучение | <img src="https://github.com/user-attachments/assets/a7d15101-d6d1-49bd-bd1d-eeb38377b38f" width="180"> |
| 7 | 275,5 | **Оптимальное** | <img src="https://github.com/user-attachments/assets/d3e99e55-b96e-417b-8b3f-a0ae580d0170" width="180"> |

</div>

Для оптимизации, вывода графика потерь и времени на обучение мы записываем функцию обучения в переменную через которую затем вытаскиваем значения потерь для валидации и обучения
Подсчет времени осуществляется через библиотеку time

Сравнение идет по трем состояниям: *недообучение* - линии идут вниз почти параллельно, *оптимальное* - линии плавно опустились и идут почти горизонтально,  валидация чуть выше и *переобучение* - после оптимального состояния линии начинают разрываться, обучение падает к нулю, а валидация начинает загибаться вверх

<div align="center">
  
## Полный график
<img src="https://github.com/user-attachments/assets/83f2fa54-71ce-43a5-9b66-fa4f2e370de8" width="500">

</div>

Быстрое обучение также связано с простой структурой модели

```python
def plot_results(history, total_time):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 6))

    # График Функция потерь (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Обучение (Training Loss)')
    plt.plot(epochs_range, val_loss, label='Валидация (Validation Loss)', linewidth=2)
    plt.title(f'Потери (Loss)\nВремя обучения: {total_time:.1f} сек.')
    plt.xlabel('Эпохи')
    plt.ylabel('Значение Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.show()
```

## Результат

![Static Badge](https://img.shields.io/badge/Accuracy-0.75-brightgreen)
![Static Badge](https://img.shields.io/badge/Loss-0.71-orange)
![Static Badge](https://img.shields.io/badge/Epochs-7-green)

Вывод результата - предсказания осуществляется через библиотеку matplotlib

```python
def predict_and_show(index):
    img = test_images[index]
    true_label = test_labels[index][0]

    # Модель ожидает массив изображений, поэтому добавляем размерность batch чтобы библиотека приняла данные для одной картинки
    img_batch = np.expand_dims(img, axis=0)

    # Получаем предсказание
    prediction = model.predict(img_batch)
    predicted_label = np.argmax(prediction[0])

    # Визуализация
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.title(f"Реально: {class_names[true_label]}\nНейросеть: {class_names[predicted_label]}")
    plt.axis('off')
    plt.show()
```

















