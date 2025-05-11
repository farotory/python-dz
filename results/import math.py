import json
import math
import os

def calculate_function(x):
    """Вычисление значения функции для варианта 3 с проверкой области определения"""
    try:
        # Проверяем, чтобы выражение под корнем было неотрицательным
        sqrt_part = 1 - 0.01 * x**2
        if sqrt_part < 0:
            return None  # Возвращаем None для значений x вне области определения
        
        return 100 * math.sqrt(sqrt_part) + 0.01 * abs(x + 10)
    except Exception as e:
        print(f"Ошибка при вычислении f({x}): {e}")
        return None

def generate_data(x_start, x_end, step):
    """Генерация данных для графика с фильтрацией недопустимых значений"""
    data = []
    x = x_start
    while x <= x_end:
        y = calculate_function(x)
        if y is not None:  # Добавляем только допустимые значения
            data.append({"x": round(x, 4), "y": round(y, 4)})
        x += step
    return data

def save_to_json(data, filename):
    """Сохранение данных в JSON файл (формат 4)"""
    result = {"data": data}
    with open(filename, 'w') as file:
        json.dump(result, file, indent=4)

def main():
    # Параметры для варианта 3
    x_start = -15
    x_end = 5
    step = 0.1  # Шаг дискретизации
    
    # Создаем директорию results, если ее нет
    os.makedirs("results", exist_ok=True)
    
    # Генерируем данные
    data = generate_data(x_start, x_end, step)
    
    # Сохраняем в файл
    output_file = os.path.join("results", "task_01_result.json")
    save_to_json(data, output_file)
    
    print(f"Данные сохранены в файл: {output_file}")
    print(f"Количество точек: {len(data)}")
    print(f"Область определения: x ∈ [-10, 10], так как 1 - 0.01x² ≥ 0 => x² ≤ 100")

if __name__ == "__main__":
    main()