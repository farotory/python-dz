import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data_from_json(file_path):
    """Загрузка данных из JSON файла"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    x_values = []
    y_values = []
    for item in data['data']:
        x_values.append(item['x'])
        y_values.append(item['y'])
    
    return x_values, y_values

def plot_graph(x_values, y_values, width=10, height=5, xmin=None, xmax=None):
    """Построение графика с указанными размерами окна и диапазоном X"""
    plt.figure(figsize=(width, height))
    plt.plot(x_values, y_values, label="f(x)", color="green")
    plt.title("График функции")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    
    current_xmin, current_xmax = plt.xlim()
    plt.xticks(np.arange(current_xmin, current_xmax + 1, 1.0))
    
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Построение графика из JSON файла')
    parser.add_argument('file', help='Путь к JSON файлу с данными')
    parser.add_argument('-w', '--width', type=float, default=10, 
                       help='Ширина окна графика (по умолчанию: 10)')
    parser.add_argument('-ht', '--height', type=float, default=5,
                       help='Высота окна графика (по умолчанию: 5)')
    parser.add_argument('--xmin', type=float, 
                       help='Минимальное значение оси X')
    parser.add_argument('--xmax', type=float, 
                       help='Максимальное значение оси X')

    args = parser.parse_args()

    try:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Файл {args.file} не найден")
            
        x, y = load_data_from_json(args.file)
        plot_graph(x, y, args.width, args.height, args.xmin, args.xmax)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == '__main__':
    main()
