import math #библиотека для математических функций
import matplotlib.pyplot as plt #библиотека для построения графиков 
import numpy as np #библиотека для работы с числовыми массивами 
import json #библиотека для json

def f(x): #объявление функции для вычисления f(x)
    term1 = 100 * math.sqrt(abs(1 - 0.01 * x**2)) #первая часть выражения
    term2 = 0.01 * abs(x + 10) #вторая часть выражения
    return term1 + term2 #возвращаем их сумма

x_values = np.arange(-15, 5, 0.1) #создаю массив значений для х от -15 до 15 с шагом 0.01
y_values = [f(x) for x in x_values] # вычисляю соответтствубщие значения функции для каждого х


# Подготовка данных для записи в JSON (только x)
data = [{"x": x} for x in x_values]
result = {"data": data}

# Запись данных в JSON-файл
with open('output.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

print("x \t f(x)") # вывожу заголовок таблицы
for x, y in zip(x_values[:100], y_values[:100]): #задаю количество точек 
    print(f"{x:.2f} \t {y:.5f}")  # Форматирование вывода

# Построение графика
plt.figure(figsize=(10, 5)) #задаю размер графика 
plt.plot(x_values, y_values, label="f(x)", color="blue") #строю график функции
plt.title("График функции f(x) при A = 0, X∈[-10;10]") #задаю заголовок
plt.xlabel("x") #подписваю оси
plt.ylabel("f(x)")
plt.grid(True) #добавляю септку для графика
plt.legend() #добавляю легенду
plt.show() #отображаю гррафик
