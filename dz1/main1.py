import numpy as np                          # Импорт NumPy для численных вычислений
import matplotlib.pyplot as plt             # Импорт Matplotlib для графиков
from objects import LayerContinuous, LayerDiscrete, Probe  # Импорт классов объектов (не используются)
import boundary                             # Импорт модуля граничных условий ABC
import tools                                # Импорт модуля анимации и визуализации

Z0 = 120.0 * np.pi                          # Волновое сопротивление вакуума η₀ ≈ 377 Ом
c0 = 299_792_458.0                          # Скорость света в вакууме, м/с
eps0 = 8.854187817e-12                      # Диэлектрическая проницаемость вакуума Ф/м

Lx = 1.5                                    # Длина области моделирования, м
eps_r = 4.0                                 # Относительная диэлектрическая проницаемость ε_r
fmin, fmax = 3e9, 8e9                       # Частотный диапазон источника 3-8 ГГц
fc = 0.5 * (fmin + fmax)                    # Центральная частота несущей f_c = 5.5 ГГц
bw = fmax - fmin                            # Ширина спектра bw = 5 ГГц
dx = Lx / 600.0                             # Шаг пространственной сетки (задается явно)
Sc = 1.0                                    # Число Куранта (CFL = 1 для максимальной стабильности)
dt = Sc * dx / c0                           # Шаг временной сетки из условия устойчивости CFL

class Sampler:                              # Класс для преобразования метров → индексы ячеек
    def __init__(self, d):                  # Инициализация размером шага d
        self.d = float(d)                   # Сохранение шага как float
    def sample(self, x):                    # Преобразование координаты x в индекс
        return int(np.floor(x / self.d + 0.5))  # Округление к ближайшей ячейке

def gauss_mod(n, t0, tau, fc, dt):          # Функция модулированного гауссова импульса
    t = n * dt                              # Текущее время t = n·dt
    g = np.exp(-((t - t0) ** 2) / (tau ** 2))  # Гауссова огибающая exp(-(t-t₀)²/τ²)
    return g * np.cos(2.0 * np.pi * fc * t) # Модуляция косинусом: g(t)×cos(2πf_ct)

class SimpleTFSF:                           # Класс TF/SF источника (Total/Scattered Field)
    def __init__(self, pos, t0, tau, fc, dt):  # Инициализация позицией и параметрами импульса
        self.pos = pos                       # Позиция источника (индекс ячейки)
        self.t0 = t0                         # Задержка гауссианы
        self.tau = tau                       # Ширина гауссианы
        self.fc = fc                         # Несущая частота
        self.dt = dt                         # Шаг времени
    
    def getH(self, n):                      # Значение источника для поля H (на шаге t+½dt)
        return -gauss_mod(n + 0.5, self.t0, self.tau, self.fc, self.dt)  # Минус для H
    
    def getE(self, n):                      # Значение источника для поля E (на шаге t)
        return gauss_mod(n, self.t0, self.tau, self.fc, self.dt)  # Прямо для E

sx = Sampler(dx)                            # Самплер для пространственной сетки
st = Sampler(dt)                            # Самплер для временной сетки

maxTime_s = 12e-9                           # Общее время моделирования, с
maxTime = st.sample(maxTime_s)              # Количество временных шагов
nx_cells = int(np.ceil(Lx / dx))            # Количество пространственных ячеек (Lx/dx)
maxSize = nx_cells                          # Размер сетки E_z
if maxSize < 3: 
    raise ValueError("Сетка слишком мелкая: maxSize должно быть >= 3")  # Проверка сетки

sourcePos = sx.sample(Lx * 0.01)            # Позиция источника: 1% от Lx от левого края
probePos = sx.sample(2 * Lx / 3)            # Позиция пробника: 2/3 от Lx
probes = [Probe(probePos, maxTime)]         # Создание списка пробников (1 шт.)

eps = np.ones(maxSize) * eps_r              # Относительная ε_r = 4.0 во всех ячейках
mu = np.ones(maxSize - 1) * 1.0             # Относительная μ_r = 1.0 везде
sigma = np.zeros(maxSize)                   # Проводимость σ = 0 (без потерь)

loss = (sigma * dt) / (2.0 * eps * eps0)    # Коэффициент диэлектрических потерь
ceze = (1.0 - loss) / (1.0 + loss)          # Yee-коэффициент: E←E (учет диэлектрика)
cezh = Z0 / (eps * (1.0 + loss))            # Yee-коэффициент: E←∇×H (учет диэлектрика)

Ez = np.zeros(maxSize)                      # Электрическое поле E_z (N узлов)
Hy = np.zeros(maxSize - 1)                  # Магнитное поле H_y (N-1 узлов между E)

left_bc = boundary.ABCSecondLeft(eps[0], mu[0], Sc)  # ABC 2-го порядка слева
right_bc = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)  # ABC 2-го порядка справа

t0 = 3.0 / fmin                            # Задержка гауссианы t₀ = 3/f_min
tau = 1.0 / bw                             # Ширина гауссианы τ = 1/bw
source = SimpleTFSF(sourcePos, t0, tau, fc, dt)  # Создание TF/SF источника

speed_refresh = 15                          # Обновление анимации каждые 15 шагов
display_field = Ez                          # Поле для отображения (E_z)
display_ylabel = 'Ez, В/м'                  # Подпись оси Y
display_ymin = -200                         # Нижний предел анимации
display_ymax = 200                          # Верхний предел анимации

display = tools.AnimateFieldDisplay(        # Создание окна анимации
    dx, dt, maxSize,                        # Параметры сетки
    display_ymin, display_ymax,             # Пределы поля
    display_ylabel,                         # Подпись Y
    title='fdtd_variant3'                   # Заголовок
)
display.activate()                          # Активация анимации
display.drawSources([sourcePos])            # Отрисовка позиции источника
display.drawProbes([probePos])              # Отрисовка позиции пробника

for t in range(1, maxTime):                 # Основной цикл FDTD (временные шаги)
    Hy += (Ez[1:] - Ez[:-1]) * (Sc / (Z0 * mu))  # H←∇×E (дискретный ротор)
    if 0 < sourcePos < len(Hy) + 1:         # Проверка границ для источника H
        Hy[sourcePos - 1] += source.getH(t) # Добавление источника в H

    Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy[:-1])  # E[1:-1]←ceze·E+cezh·∇×H
    if 0 <= sourcePos < len(Ez):            # Проверка границ для источника E
        Ez[sourcePos] += source.getE(t)     # Добавление источника в E

    left_bc.updateField(Ez, Hy)             # Обновление левой ABC границы
    right_bc.updateField(Ez, Hy)            # Обновление правой ABC границы

    for p in probes:                        # Запись данных во все пробники
        p.addData(Ez, Hy)

    if (t % speed_refresh) == 0:            # Обновление анимации (каждые 15 шагов)
        display.updateData(display_field, t)

display.stop()                              # Остановка анимации

probe = probes[0]                           # Получение данных первого пробника
t_axis = np.arange(len(probe.E)) * dt       # Временная ось t = [0, dt, 2dt, ...]
signal = np.array(probe.E)                  # Сигнал E_z(t) из пробника

plt.rcParams.update({                       # Настройка стиля графиков Matplotlib
    "figure.figsize": (9, 4),               # Размер фигуры
    "font.size": 12,                        # Размер шрифта
    "lines.linewidth": 1.8,                 # Толщина линий
    "axes.grid": True,                      # Сетка на графиках
})

plt.figure()                                # Создание фигуры для временного сигнала
plt.plot(t_axis * 1e9, signal)              # График E_z(t) в нс
plt.xlabel("t, нс")                         # Подпись оси X
plt.ylabel("Ez, В/м")                       # Подпись оси Y
plt.title("Сигнал на датчике")              # Заголовок
plt.xlim(0, maxTime_s * 1e9)                # Ограничение по X (0-12 нс)
plt.tight_layout()                          # Оптимизация макета

window = np.hanning(len(signal))            # Hanning-окно против спектральной утечки
spec = np.fft.fft(signal * window)          # FFT сигнала с окном
freq = np.fft.fftfreq(len(signal), dt)      # Частотная ось FFT
mask_f = freq >= 0                          # Маска положительных частот
freq_pos = freq[mask_f]                     # Только положительные частоты
spec_amp = np.abs(spec[mask_f])             # Модуль спектра
if spec_amp.max() != 0:                     # Нормировка (если спектр не нулевой)
    spec_amp /= spec_amp.max()

plt.figure()                                # Создание фигуры для спектра
plt.plot(freq_pos * 1e-9, spec_amp)         # График нормированного спектра
plt.xlabel("f, ГГц")                        # Подпись оси X
plt.ylabel("|S(f)| / max")                  # Подпись оси Y
plt.title("Нормированный спектр сигнала")   # Заголовок
plt.xlim(0, 10)                             # Ограничение 0-10 ГГц
plt.tight_layout()                          # Оптимизация макета
plt.show()                                  # Показать все графики
