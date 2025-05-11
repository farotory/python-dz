import argparse
import json
import matplotlib.pyplot as plt
import sys

def load_data_from_json(filename):
    """Загрузка данных из JSON файла (формат 4)"""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            x = [item['x'] for item in data['data']]
            y = [item['y'] for item in data['data']]
            
            if not x or not y:
                raise ValueError("Файл не содержит данных")
                
            return x, y
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Ошибка: Файл '{filename}' не является корректным JSON")
        sys.exit(1)
    except KeyError as e:
        print(f"Ошибка: В файле отсутствует необходимый ключ - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)

def plot_graph(x, y, xmin=None, xmax=None):
    """Построение графика с возможностью ограничения по оси X"""
    plt.figure(figsize=(12, 7))
    
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('График функции: y = 100√(1-0.01x²) + 0.01|x+10|', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Добавляем подписи к осям
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Построение графика функции из JSON файла (формат 4)',
        epilog='Пример использования: python task_02.py input.json --xmin=-10 --xmax=10'
    )
    parser.add_argument('filename', help='Имя JSON файла с данными')
    parser.add_argument('--xmin', type=float, help='Минимальное значение по оси X')
    parser.add_argument('--xmax', type=float, help='Максимальное значение по оси X')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    x, y = load_data_from_json(args.filename)
    plot_graph(x, y, args.xmin, args.xmax)

if __name__ == '__main__':
    main()