import json  # библиотека для работы с JSON-файлами
import math
import matplotlib.pyplot as plt
import numpy as np  # библиотека для численных операций с массивами
import argparse  # библиотека для обработки аргументов командной строки

def f(x):
    term1 = 100 * math.sqrt(abs(1 - 0.01 * x**2))
    term2 = 0.01 * abs(x + 10)
    return term1 + term2

def load_data(file_path):  # функция для загрузки данных из JSON-файла
    with open(file_path, 'r') as file:  # открываем файл в режиме чтения
        data = json.load(file)  # Загружаем данные из JSON-файла
    return data['data']  # возвращаем только массив данных (по ключу 'data')

    x_values = np.array([point['x'] for point in data])  # извлекаем значения x из данных
    y_values = np.array([f(point['x']) for point in data])  # Вычисляем значения y для каждого x с помощью функции f(x)

    plt.figure(figsize=(10, 5))  # создаем новую фигуру размером 10x5 
    plt.plot(x_values, y_values, label="f(x)", color="blue")  # строим график

    if title:  # устанавливаем заголовок графика, если он задан
        plt.title(title)
    if xlabel:  # устанавливаем подпись оси X, если она задана
        plt.xlabel(xlabel)
    if ylabel:  # устанавливаем подпись оси Y, если она задана
        plt.ylabel(ylabel)
    if grid:  # включаем сетку, если параметр grid=True
        plt.grid(True)
    
    # Устанавливаем пределы оси X, если они заданы
    if xmin:
        plt.xlim(xmin, xmax)

    plt.legend()  # добавляем легенду
    plt.show()  # отображаем график

def main():
    parser = argparse.ArgumentParser(description='Plot graph from JSON data.')  # Создаем парсер аргументов командной строки
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')  # Добавляем аргументы:
    # Обязательный аргумент - путь к входному файлу
    # Необязательные аргументы:
    parser.add_argument('--title', type=str, help='Title of the graph')
    parser.add_argument('--xlabel', type=str, help='Label for the X axis')
    parser.add_argument('--ylabel', type=str, help='Label for the Y axis')
    parser.add_argument('--grid', action='store_true', help='Show grid on the graph')
    parser.add_argument('--x_step', type=float, default=1.0, help='Step for X axis ticks')
    parser.add_argument('--xmin', type=float, help='Minimum value for X axis')
    parser.add_argument('--xmax', type=float, help='Maximum value for X axis')

    args = parser.parse_args()  # парсим аргументы командной строки

    data = load_data(args.input_file)  # загружаем данные из файла
    plot_graph(data, args.title, args.xlabel, args.ylabel, args.grid, args.x_step, args.xmin, args.xmax)  # строим график с заданными параметрами

if __name__ == '__main__':  # точка входа в программу
    main()
    #команда для запуска: python laba2.py work2.json --title "My Graph" --xlabel "X Axis" --ylabel "Y Axis" --grid --xmin -20 --xmax 20
    