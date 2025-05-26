# Импортируем необходимые библиотеки
import numpy as np  # Для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # Для построения графиков
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-визуализации

# Определяем функцию Экли (Ackley function) - тестовая функция для оптимизации
def ackley_function(x):
    x = np.array(x)  # Преобразуем входные данные в массив NumPy
    D = len(x)  # Получаем размерность пространства (количество переменных)
    sum_sq = np.sum(x**2)  # Сумма квадратов переменных
    sqrt_term = np.sqrt(sum_sq / D)  # Среднеквадратичное значение
    term1 = -20 * np.exp(-0.02 * sqrt_term)  # Первое слагаемое функции
    
    cos_term = np.sum(np.cos(2 * np.pi * x))  # Сумма косинусов
    term2 = -np.exp(cos_term / D)  # Второе слагаемое функции
    
    return term1 + term2 + 20 + np.e  # Возвращаем значение функции (с добавлением констант)

# Параметры для построения графиков
x10, x20 = 0.0, 0.0  # Тестовая точка (0.0; 0.0) - обычно глобальный минимум
x1_min, x1_max = -5.0, 5.0  # Интервал для переменной x1
x2_min, x2_max = -5.0, 5.0  # Интервал для переменной x2
num_points = 100  # Количество точек для построения графиков

# Создаем сетку для 3D графиков
x1 = np.linspace(x1_min, x1_max, num_points)  # Равномерно распределенные точки по x1
x2 = np.linspace(x2_min, x2_max, num_points)  # Равномерно распределенные точки по x2
X1, X2 = np.meshgrid(x1, x2)  # Создаем координатную сетку
Z = np.zeros_like(X1)  # Матрица для значений функции

# Вычисляем значения функции для каждой точки сетки
for i in range(num_points):
    for j in range(num_points):
        Z[i, j] = ackley_function([X1[i, j], X2[i, j]])

# Значение функции в тестовой точке
test_point_value = ackley_function([x10, x20])

# Создаем фигуру c 4 подграфиками (2x2)
fig = plt.figure(figsize=(16, 12))  # Задаем размер фигуры
# Общий заголовок для всех графиков
fig.suptitle(f'Функция Экли на интервалах x1∈[{x1_min}, {x1_max}], x2∈[{x2_min}, {x2_max}]\n'
             f'Значение в точке ({x10}, {x20}) = {test_point_value:.2f}', fontsize=14)

# 1. 3D поверхность в изометрическом виде
ax1 = fig.add_subplot(221, projection='3d')  # Первый подграфик
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)  # Поверхность
# Отмечаем тестовую точку
ax1.scatter([x10], [x20], [test_point_value], color='red', s=100, label=f'({x10}, {x20})')
ax1.set_title('3D поверхность')  # Заголовок
ax1.set_xlabel('x1')  # Подпись оси X
ax1.set_ylabel('x2')  # Подпись оси Y
ax1.set_zlabel('f(x1,x2)')  # Подпись оси Z
ax1.legend()  # Легенда
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)  # Цветовая шкала

# 2. 3D поверхность, вид сверху
ax2 = fig.add_subplot(222, projection='3d')  # Второй подграфик
surf2 = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)  # Поверхность
ax2.scatter([x10], [x20], [test_point_value], color='red', s=100, label=f'({x10}, {x20})')
ax2.set_title('Вид сверху')  # Заголовок
ax2.set_xlabel('x1')  # Подпись оси X
ax2.set_ylabel('x2')  # Подпись оси Y
ax2.set_zlabel('f(x1,x2)')  # Подпись оси Z
ax2.view_init(elev=90, azim=0)  # Устанавливаем вид сверху (elevation=90 градусов)
ax2.legend()  # Легенда
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)  # Цветовая шкала

# 3. График сечения при x1 = x10 (фиксируем x1, меняем x2)
ax3 = fig.add_subplot(223)  # Третий подграфик
# Строим график зависимости функции от x2 при фиксированном x1=x10
ax3.plot(x2, [ackley_function([x10, x2i]) for x2i in x2])
# Отмечаем тестовую точку
ax3.scatter(x20, test_point_value, color='red', s=100, 
           label=f'x2={x20}, f={test_point_value:.2f}')
ax3.set_title(f'Сечение при x1 = {x10}')  # Заголовок
ax3.set_xlabel('x2')  # Подпись оси X
ax3.set_ylabel(f'f({x10}, x2)')  # Подпись оси Y
ax3.grid(True)  # Включаем сетку
ax3.legend()  # Легенда

# 4. График сечения при x2 = x20 (фиксируем x2, меняем x1)
ax4 = fig.add_subplot(224)  # Четвертый подграфик
# Строим график зависимости функции от x1 при фиксированном x2=x20
ax4.plot(x1, [ackley_function([x1i, x20]) for x1i in x1])
# Отмечаем тестовую точку
ax4.scatter(x10, test_point_value, color='red', s=100, 
           label=f'x1={x10}, f={test_point_value:.2f}')
ax4.set_title(f'Сечение при x2 = {x20}')  # Заголовок
ax4.set_xlabel('x1')  # Подпись оси X
ax4.set_ylabel(f'f(x1, {x20})')  # Подпись оси Y
ax4.grid(True)  # Включаем сетку
ax4.legend()  # Легенда

plt.tight_layout()  # Автоматическая настройка расположения подграфиков
plt.show()  # Отображаем все графики