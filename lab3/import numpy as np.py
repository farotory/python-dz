import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ackley_function(x):
    
    x = np.array(x)
    D = len(x)
    sum_sq = np.sum(x**2)
    sqrt_term = np.sqrt(sum_sq / D)
    term1 = -20 * np.exp(-0.02 * sqrt_term)
    
    cos_term = np.sum(np.cos(2 * np.pi * x))
    term2 = -np.exp(cos_term / D)
    
    return term1 + term2

# Параметры для построения графиков
x10, x20 = 0.0, 0.0  # Тестовая точка (0.0; 0.0)
x1_min, x1_max = -5.0, 5.0  # Интервал для x1
x2_min, x2_max = -5.0, 5.0  # Интервал для x2
num_points = 100

# Создаем сетку для 3D графиков
x1 = np.linspace(x1_min, x1_max, num_points)
x2 = np.linspace(x2_min, x2_max, num_points)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

# Вычисляем значения функции для каждой точки сетки
for i in range(num_points):
    for j in range(num_points):
        Z[i,j] = ackley_function([X1[i,j], X2[i,j]])

# Значение функции в тестовой точке
test_point_value = ackley_function([x10, x20])

# Создаем фигуру с 4 подграфиками
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'Функция на интервалах x1∈[{x1_min}, {x1_max}], x2∈[{x2_min}, {x2_max}]\n'
             f'Значение в точке ({x10}, {x20}) = {test_point_value:.2f}', fontsize=14)

# 1. 3D поверхность в изометрическом виде
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax1.scatter([x10], [x20], [test_point_value], color='red', s=100, label=f'({x10}, {x20})')
ax1.set_title(f'3D поверхность\nx1∈[{x1_min}, {x1_max}], x2∈[{x2_min}, {x2_max}]')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1,x2)')
ax1.legend()
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 2. 3D поверхность, вид сверху
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax2.scatter([x10], [x20], [test_point_value], color='red', s=100, label=f'({x10}, {x20})')
ax2.set_title(f'Вид сверху\nx1∈[{x1_min}, {x1_max}], x2∈[{x2_min}, {x2_max}]')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('f(x1,x2)')
ax2.view_init(elev=90, azim=0)  # Вид сверху
ax2.legend()
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# 3. График при x1 = x2 (линия через точку (x10, x20))
t = np.linspace(max(x1_min, x2_min), min(x1_max, x2_max), num_points)
line_values = np.array([ackley_function([ti, ti]) for ti in t])

ax3 = fig.add_subplot(223)
ax3.plot(t, line_values)
ax3.scatter(0, test_point_value, color='red', s=100, label=f'x=0, f={test_point_value:.2f}')
ax3.set_title(f'График при x1 = x2\nИнтервал [{max(x1_min, x2_min):.1f}, {min(x1_max, x2_max):.1f}]')
ax3.set_xlabel('x (x1 = x2)')
ax3.set_ylabel('f(x,x)')
ax3.grid(True)
ax3.legend()

# 4. Контурный график с выделенной точкой (x10, x20)
ax4 = fig.add_subplot(224)
contour = ax4.contourf(X1, X2, Z, levels=20, cmap='viridis')
ax4.plot(x10, x20, 'ro', markersize=10, label=f'({x10}, {x20})')
ax4.set_title(f'Контурный график\nx1∈[{x1_min}, {x1_max}], x2∈[{x2_min}, {x2_max}]')
ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.annotate(f'({x10}, {x20})\nf={test_point_value:.2f}', 
             (x10, x20), 
             textcoords="offset points", 
             xytext=(10,10), 
             ha='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
ax4.legend()
fig.colorbar(contour, ax=ax4, shrink=0.8, aspect=10)

plt.tight_layout()
plt.show()
