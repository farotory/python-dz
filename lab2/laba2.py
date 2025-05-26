# Импортируем необходимые библиотеки:
import json              # Для работы с JSON-файлами
import matplotlib.pyplot as plt  # Для построения графиков
import numpy as np       # Для работы с числовыми массивами
import argparse          # Для обработки аргументов командной строки
import os                # Для работы с файловой системой

def load_data_from_json(file_path):
    """Загрузка данных из JSON файла"""
    with open(file_path, 'r') as file:  # Открываем файл в режиме чтения
        data = json.load(file)  # Загружаем JSON-данные в словарь
    
    # Создаём пустые списки для хранения значений x и y
    x_values = []
    y_values = []
    
    # Проходим по всем элементам в data['data'] и извлекаем x и y
    for item in data['data']:
        x_values.append(item['x'])
        y_values.append(item['y'])
    
    return x_values, y_values  # Возвращаем списки значений

def plot_graph(x_values, y_values, width=10, height=5, xmin=None, xmax=None):
    """Построение графика с указанными размерами окна и диапазоном X"""
    plt.figure(figsize=(width, height))  # Устанавливаем размер графика
    
    # Рисуем график:
    plt.plot(x_values, y_values, label="f(x)", color="green")
    plt.title("График функции")  # Заголовок
    plt.xlabel("x")  # Подпись оси X
    plt.ylabel("f(x)")  # Подпись оси Y
    plt.grid(True)  # Включаем сетку
    
    # Если заданы xmin и xmax, устанавливаем границы оси X
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    
    # Получаем текущие границы оси X и устанавливаем метки с шагом 1.0
    current_xmin, current_xmax = plt.xlim()
    plt.xticks(np.arange(current_xmin, current_xmax + 1, 1.0))
    
    plt.legend()  # Показываем легенду
    plt.show()    # Отображаем график

def main():
    # Настраиваем парсер аргументов командной строки:
    parser = argparse.ArgumentParser(description='Построение графика из JSON файла')
    
    # Добавляем аргументы:
    parser.add_argument('file', help='Путь к JSON файлу с данными')
    parser.add_argument('-w', '--width', type=float, default=10, 
                       help='Ширина окна графика (по умолчанию: 10)')
    parser.add_argument('-ht', '--height', type=float, default=5,
                       help='Высота окна графика (по умолчанию: 5)')
    parser.add_argument('--xmin', type=float, 
                       help='Минимальное значение оси X')
    parser.add_argument('--xmax', type=float, 
                       help='Максимальное значение оси X')

    args = parser.parse_args()  # Разбираем аргументы

    try:
        # Проверяем, существует ли файл
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Файл {args.file} не найден")
            
        # Загружаем данные и строим график
        x, y = load_data_from_json(args.file)
        plot_graph(x, y, args.width, args.height, args.xmin, args.xmax)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")  # Выводим ошибку, если что-то пошло не так

if __name__ == '__main__':
    main()  # Запускаем программу, только если скрипт выполняется напрямую