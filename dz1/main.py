# main.py — вариант 3 (L=1.5 м, eps_r=4, модулированный гаусс 3–8 ГГц)
# Улучшенная, откомментированная версия скрипта FDTD.
import numpy as np
import matplotlib.pyplot as plt

# Предполагаемые локальные модули: objects, boundary, tools
# В objects ожидаются классы: LayerContinuous, LayerDiscrete, Probe
# (в оригинале был опечаточный импорт "Probed" и использование "Probe")
from objects import LayerContinuous, LayerDiscrete, Probe
import boundary
import tools

# --- физические константы ---
Z0 = 120.0 * np.pi             # волновое сопротивление свободного пространства (~377 Ом)
c0 = 299_792_458.0             # скорость света, м/с
eps0 = 8.854187817e-12         # электрическая постоянная (пермиттивность вакуума)

# --- параметры варианта 3 ---
Lx = 1.5                       # длина расчетной области вдоль оси X, м
eps_r = 4.0                    # относительная диэл. проницаемость материала
fmin, fmax = 3e9, 8e9          # полоса возбуждения: от 3 ГГц до 8 ГГц
fc = 0.5 * (fmin + fmax)       # центральная частота
bw = fmax - fmin               # ширина полосы

# --- сетка (пространственная и временная) ---
nx_cells = 600                 # целевое число ячеек вдоль оси X (можно изменить)
dx = Lx / nx_cells             # шаг сетки по x, м

Sc = 0.99                      # число Курранта (Courant) — должен быть < 1 для стабильности
dt = Sc * dx / c0              # временной шаг, с

# Утилита для перевода размеров в индексы сетки
class Sampler:
    def __init__(self, d): self.d = float(d)
    def sample(self, x):   # возвращает ближайший индекс (int) по позиции x (м)
        return int(np.floor(x / self.d + 0.5))

sx = Sampler(dx)
st = Sampler(dt)

maxTime_s = 12e-9               # максимальное время моделирования в секундах
maxTime = st.sample(maxTime_s) # число временных шагов (integer)
maxSize = sx.sample(Lx)        # число узлов (точек) в пространственной сетке

# Контроль — чтобы массивы имели не нулевой размер
if maxSize < 3:
    raise ValueError("Сетка слишком мелкая: maxSize должно быть >= 3")

# --- положения источника и пробника (в индексах сетки) ---
sourcePos = sx.sample(Lx*0.01)          # источник в 1/3 от начала
probePos = sx.sample(2 * Lx / 3)       # пробник в 2/3 от начала

# Создаем объект-датчик (Probe) — ожидаем, что класс Probe( pos, maxTime ) есть в objects.py
probes = [Probe(probePos, maxTime)]

# --- слой: задаем однотипную среду по всей длине ---
layer_cont = LayerContinuous(0.0, xmax=Lx, eps=eps_r, sigma=0.0)  # континуальная модель
# Перевод в дискретный слой (индексы сетки)
layer = LayerDiscrete(
    sx.sample(layer_cont.xmin),
    sx.sample(layer_cont.xmax),
    layer_cont.eps,
    1.0,                # mu_r (здесь предполагаем mu_r = 1.0)
    layer_cont.sigma
)

# --- массивы параметров среды, приведённые к размерам сетки ---
# eps — относительная проницаемость в каждой точке (N точек)
eps = np.ones(maxSize) * layer.eps

# mu — относительная магнитная проницаемость на промежутках Hy (N-1)
mu = np.ones(maxSize - 1) * layer.mu

# sigma — электрическая проводимость в каждой точке (здесь нулевая в задаче)
sigma = np.zeros(maxSize)

# --- коэффициенты обновления (учёт потерь через sigma) ---
# loss — безразмерный параметр рассеяния в формуле (используется для модификации коэффициентов)
# (в некоторых реализациях loss = (sigma * dt) / (2 * eps * eps0))
loss = (sigma * dt) / (2.0 * eps * eps0)

# ceze — коэффициент перед старым Ez (учёт диссипации)
ceze = (1.0 - loss) / (1.0 + loss)

# cezh — коэффициент перед членом разности Hy (упрощённый вид; сохранён стиль вашего кода)
# Примечание: точная форма зависит от нормировок в вашем Hy-обновлении; 
# я оставил коэффициент похожим на исходный, но нормализовал по eps.
cezh = Z0 / (eps * (1.0 + loss))

# --- поля (нулевые начальные условия) ---
Ez = np.zeros(maxSize)           # электрическое поле Ez на узлах (N)
Hy = np.zeros(maxSize - 1)       # магнитное поле Hy на промежутках (N-1)

# --- граничные условия: здесь ABC 2-го порядка слева и справа ---
# boundary.ABCSecondLeft / Right ожидают параметры среды на границе
left_bc = boundary.ABCSecondLeft(eps[0],  mu[0],  Sc)
right_bc = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

# --- источник: модулированный гауссов импульс (3–8 ГГц) ---
# задаём временные параметры импульса
t0 = 3.0 / fmin    # задержка импульса (в секундах) — выбран так, как у вас
tau = 1.0 / bw     # временная ширина огибающей (приближённо)

def gauss_mod(n):
    """
    Возвращает значение модулированного гаусс-импульса в момент n (целый шаг).
    g(t) = exp(-((t - t0)^2)/tau^2) * cos(2π f_c t)
    """
    t = n * dt
    g = np.exp(-((t - t0) ** 2) / (tau ** 2))
    return g * np.cos(2.0 * np.pi * fc * t)

# Простая оболочка TFSF-источника с интерфейсами getH(n) и getE(n)
class SimpleTFSF:
    def __init__(self, pos):
        self.pos = pos
    def getH(self, n):   # возбуждение для Hy (инжектируется в Hy[sourcePos-1])
        return -gauss_mod(n + 0.5)   # сдвиг на полшага для H
    def getE(self, n):   # возбуждение для Ez (инжектируется в Ez[sourcePos])
        return gauss_mod(n)

source = SimpleTFSF(sourcePos)

# --- параметры анимации / отображения поля (опционально используемые tools.Display) ---
speed_refresh = 15               # обновлять визуализацию каждые speed_refresh шагов
display_field = Ez
display_ylabel = 'Ez, В/м'
display_ymin = -200
display_ymax =  200

# Инструмент отображения (ожидается, что tools.AnimateFieldDisplay реализован)
display = tools.AnimateFieldDisplay(
    dx, dt,
    maxSize,
    display_ymin, display_ymax,
    display_ylabel,
    title='fdtd_variant3'
)
display.activate()
display.drawSources([sourcePos])     # показать позицию источника на картинке
display.drawProbes([probePos])       # показать позицию пробника
display.drawBoundary(layer.xmin)     # показать границы слоя (xmin)
display.drawBoundary(layer.xmax)     # показать границы слоя (xmax)

# --- основной FDTD-цикл по времени ---
# Обновление полей в явной схеме Yee (1D): сначала H, затем E
for t in range(1, maxTime):
    # --- обновление H (Hy на промежутках) ---
    # Hy <- Hy + (Ez[i+1] - Ez[i]) * (Sc / (Z0 * mu))
    Hy += (Ez[1:] - Ez[:-1]) * (Sc / (Z0 * mu))
    # добавляем вклад источника в поле H (инжекция в Hy[sourcePos - 1])
    # защита от выхода за границы
    if 0 < sourcePos < len(Hy) + 1:
        Hy[sourcePos - 1] += source.getH(t)

    # --- обновление E (Ez на узлах), внутренние узлы 1..N-2 ---
    # Ez[i] = ceze[i] * Ez[i] + cezh[i] * (Hy[i] - Hy[i-1])
    Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy[:-1])

    # добавляем вклад источника в поле E в узле sourcePos
    if 0 <= sourcePos < len(Ez):
        Ez[sourcePos] += source.getE(t)

    # --- граничные условия (внешние) ---
    left_bc.updateField(Ez, Hy)
    right_bc.updateField(Ez, Hy)

    # --- запись в пробники (датчики) ---
    for p in probes:
        p.addData(Ez, Hy)

    # --- анимация (обновление визуализации) ---
    if (t % speed_refresh) == 0:
        display.updateData(display_field, t)

display.stop()   # остановить/закрыть отображение (если требуется)

# --- постобработка: временной сигнал и его спектр ---
probe = probes[0]
t_axis = np.arange(len(probe.E)) * dt   # временная ось (опираемся на накопленные данные датчика)
signal = np.array(probe.E)              # временной сигнал Ez в пробнике

# Параметры оформления графиков
plt.rcParams.update({
    "figure.figsize": (9, 4),
    "font.size": 12,
    "lines.linewidth": 1.8,
    "axes.grid": True,
})

# --- полный временной сигнал ---
plt.figure()
plt.plot(t_axis * 1e9, signal)
plt.xlabel("t, нс")
plt.ylabel("Ez, В/м")
plt.title("Сигнал на датчике")
plt.xlim(0, maxTime_s * 1e9)
plt.tight_layout()

# --- спектр: окно Хэннинга + FFT ---
window = np.hanning(len(signal))
spec = np.fft.fft(signal * window)
freq = np.fft.fftfreq(len(signal), dt)
mask_f = freq >= 0
freq_pos = freq[mask_f]
spec_amp = np.abs(spec[mask_f])
# нормируем на максимальное значение для удобства сравнения
if spec_amp.max() != 0:
    spec_amp /= spec_amp.max()

plt.figure()
plt.plot(freq_pos * 1e-9, spec_amp)
plt.xlabel("f, ГГц")
plt.ylabel("|S(f)| / max")
plt.title("Нормированный спектр сигнала")
plt.xlim(0, 10)
plt.tight_layout()

plt.show()
