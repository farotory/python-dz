# Импортируем необходимые библиотеки:
import os                     # Для работы с файловой системой (создание папок)
import math                   # Для математических операций (корень, модуль)
import matplotlib.pyplot as plt  # Для построения графиков
import numpy as np            # Для работы с массивами чисел
import json                   # Для сохранения данных в JSON-формате

# Определяем функцию f(x), которая вычисляет значение по заданной формуле:
def f(x):
    return 100 * math.sqrt(  # Умножаем на 100 и берём квадратный корень от:
        abs(1 - 0.01 * x**2)  # Модуль выражения (1 - 0.01x²)
        + 0.01 * abs(x + 10)  # Плюс 0.01 * модуль (x + 10)
    )

# Создаём папку 'results', если её ещё нет (exist_ok=True предотвращает ошибку, если папка уже есть):
os.makedirs('results', exist_ok=True)

# Генерируем массив x-значений от -15 до 5 с шагом 0.01:
x_values = np.arange(-15, 5.01, 0.01)

# Вычисляем y-значения для каждого x, применяя функцию f(x):
y_values = [f(x) for x in x_values]

# Собираем данные в список словарей вида [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]:
data = [{"x": x, "y": y} for x, y in zip(x_values, y_values)]

# Создаём итоговый словарь с ключом "data", содержащий все пары (x, y):
result = {"data": data}

# Задаём путь к файлу output.json внутри папки results:
output_path = os.path.join('results', 'output.json')

# Сохраняем данные в JSON-файл с отступами (indent=4 для красивого форматирования):
with open(output_path, 'w') as json_file:
    json.dump(result, json_file, indent=4)

# Выводим путь к сохранённому файлу:
print("Файл сохранён в:", output_path)

# Выводим первые 100 значений x и y в виде таблицы (для демонстрации):
print("x \t y = f(x)")
for x, y in zip(x_values[:100], y_values[:100]):
    print(f"{x:.2f} \t {y:.5f}")  # Форматируем вывод: x с 2 знаками, y с 5

# Настраиваем график:
plt.figure(figsize=(10, 5))  # Задаём размер графика (ширина, высота)
plt.plot(x_values, y_values, label="y = f(x)", color="green")  # Рисуем линию графика
plt.title("График функции y = 100 * sqrt(|1 - 0.01x²| + 0.01|x + 10|), x∈[-15;5]")  # Заголовок
plt.xlabel("x")  # Подпись оси X
plt.ylabel("y")  # Подпись оси Y
plt.grid(True)   # Включаем сетку
plt.legend()     # Показываем легенду (label="y = f(x)")
plt.show()       # Отображаем график