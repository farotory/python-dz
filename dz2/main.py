import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt

# =====================================================
# КЛАСС ИСТОЧНИКА - ГАУССОВ ДИФФЕРЕНЦИРОВАННЫЙ ИМПУЛЬС
# =====================================================
class GaussianDiff:
    """Гауссов импульс для источника - создает широкополосный сигнал"""
    def __init__(self, dt, A, F, Nl, Sc=1.0, eps=1.0, mu=1.0):
        # Сохранение параметров: A=амплитуда, F=центр. частота, Nl=период, dt=шаг времени
        self.A = A
        self.F = F
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.dt = dt

    def getE(self, m, q):
        # Вычисление гауссова импульса: sin(модуляция) * exp(-гауссова оболочка)
        w = 2 * np.sqrt(np.log(self.A)) / (np.pi * self.F)      # Ширина в частотной области
        d = w * np.sqrt(np.log(self.A))                          # Задержка импульса
        dt = self.dt
        # Формула: модулированный гауссов импульс с учетом дисперсии среды
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - d / dt) / (w / dt)) ** 2))

# =====================================================
# ВСПOMOГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================
def create_layer_boundaries(layer_x_start, d1, d2, d3, d):
    """Перевод физических координат границ слоев в индексы дискретной сетки"""
    layer1_x = layer_x_start                    # Начало 1-го слоя
    layer2_x = layer1_x + d1                    # Граница 1-2 слоя
    layer3_x = layer2_x + d2                    # Граница 2-3 слоя  
    layer4_x = layer3_x + d3                    # Граница 3-4 слоя
    
    # Округление до ближайшего индекса сетки: ceil(x/d)
    layer1_DX = int(np.ceil(layer1_x / d))
    layer2_DX = int(np.ceil(layer2_x / d))
    layer3_DX = int(np.ceil(layer3_x / d))
    layer4_DX = int(np.ceil(layer4_x / d))
    
    return [layer1_DX, layer2_DX, layer3_DX, layer4_DX]

def setup_abc_coefficients(Sc, mu, eps):
    """Вычисление коэффициентов ABC (Mur 2nd order) для поглощающих границ"""
    # ЛЕВАЯ ГРАНИЦА (i=0)
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])           # Нормированный импеданс
    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)        # Коэффициенты рекуррентной формулы ABC
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)
    
    # ПРАВАЯ ГРАНИЦА (i=maxSize-1)
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)
    
    return (k1Left, k2Left, k3Left, k4Left, k1Right, k2Right, k3Right, k4Right)

# =====================================================
# ОСНОВНАЯ ПРОГРАММА - 1D FDTD СИМУЛЯЦИЯ
# =====================================================
if __name__ == '__main__':
    # ==================== ПАРАМЕТРЫ ДИСКРЕТИЗАЦИИ ====================
    d = 0.3e-3               # Шаг пространственной сетки, м (0.3 мкм)
    Z0 = 120.0 * np.pi       # Характерное сопротивление свободного пространства
    Sc = 1.0                 # Коэффициент Кори (стабильность)
    mu0 = np.pi * 4e-7       # Магн. прониц. вакуума
    eps0 = 8.854187817e-12   # Электр. прониц. вакуума
    c = 1.0 / np.sqrt(mu0 * eps0)  # Скорость света

    dt = d / c * Sc          # Шаг по времени (условие Кори)
    print(f"dt = {dt}")
    
    maxTime_sec = 25e-9      # Общее время симуляции, с
    maxTime = int(np.ceil(maxTime_sec / dt))     # Кол-во шагов по времени
    sizeX_m = 1              # Длина области, м
    maxSize = int(np.ceil(sizeX_m / d))          # Кол-во узлов сетки

    # ==================== ПАРАМЕТРЫ СЛОЕВ ====================
    d1, d2, d3 = 0.06, 0.06, 0.10     # Толщины слоев, м
    eps1, eps2, eps3, eps4 = 3.5, 2.2, 2.0, 6.0  # Отн. диэльтр. прониц-ти
    layer_x_start = 0.52              # Начало слоев, м

    # Вычисление индексов границ слоев
    layer_boundaries = create_layer_boundaries(layer_x_start, d1, d2, d3, d)
    layer1_DX, layer2_DX, layer3_DX, layer4_DX = layer_boundaries

    # ==================== ИСТОЧНИК И ПРОБЫ ====================
    sourcePosm = 0.1                 # Позиция источника, м
    sourcePos = int(np.ceil(sourcePosm / d))     # Индекс источника
    probesPos = [int(np.ceil(0.1 / d)), int(np.ceil(0.01 / d))]  # Позиции проб
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]    # Создание детекторов

    # ==================== ПРОФИЛЬ СРЕДЫ ====================
    eps = np.ones(maxSize)           # Диэльтр. прониц-ть (1=вакуум)
    eps[layer1_DX:] = eps1           # Слой 1: ε=3.5
    eps[layer2_DX:] = eps2           # Слой 2: ε=2.2  
    eps[layer3_DX:] = eps3           # Слой 3: ε=2.0
    eps[layer4_DX:] = eps4           # Слой 4: ε=6.0
    mu = np.ones(maxSize - 1)        # Магн. прониц-ть (1=немагнитная)

    # ==================== ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ====================
    Ez = np.zeros(maxSize)           # Электрическое поле Ez (Yee: на узлах)
    Ezspectrumpad = np.zeros(maxTime) # Спектр падающей волны
    Ezspectrumotr = np.zeros(maxTime) # Спектр отраженной волны
    Hy = np.zeros(maxSize - 1)       # Магнитное поле Hy (Yee: между узлами)

    # ==================== ИСТОЧНИК ====================
    source = GaussianDiff(dt, 50, 5e9, 455)  # A=50, F=5ГГц, Nl=455

    # ==================== ABC КОЭФФИЦИЕНТЫ ====================
    (k1Left, k2Left, k3Left, k4Left, 
     k1Right, k2Right, k3Right, k4Right) = setup_abc_coefficients(Sc, mu, eps)

    # История значений для ABC (3 предыдущих шага)
    oldEzLeft1 = np.zeros(3)
    oldEzLeft2 = np.zeros(3)
    oldEzRight1 = np.zeros(3)
    oldEzRight2 = np.zeros(3)

    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin, display_ymax = -1.1, 1.1

    # Анимация поля
    display = tools.AnimateFieldDisplay(maxSize, display_ymin, display_ymax,
                                        display_ylabel, d, dt)
    display.activate()
    display.drawProbes(probesPos)        # Отметить пробы
    display.drawSources([sourcePos])     # Отметить источник
    display.drawBoundary(layer1_DX)      # Границы слоев
    display.drawBoundary(layer2_DX)
    display.drawBoundary(layer3_DX)
    display.drawBoundary(layer4_DX)

    # ==================== ОСНОВНОЙ ЦИКЛ FDTD (Yee scheme) ====================
    for q in range(1, maxTime):
        # ШАГ 1: Обновление H-поля (curl E → ∂H/∂t)
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)           # Основное уравнение
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)  # Источник в H

        # ШАГ 2: Обновление E-поля (curl H → ∂E/∂t)  
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]  # Основное уравнение
        # Hard source в электрическом поле (сдвиг на полшага)
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        # ШАГ 3: ABC ЛЕВАЯ ГРАНИЦА (Mur 2nd order)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +           # Рекуррентная формула
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])
        oldEzLeft2[:] = oldEzLeft1[:]       # Сдвиг истории (t-1 → t-2)
        oldEzLeft1[:] = Ez[0:3]             # Текущее → t-1

        # ШАГ 4: ABC ПРАВАЯ ГРАНИЦА (Mur 2nd order)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +    # Рекуррентная формула
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])
        oldEzRight2[:] = oldEzRight1[:]     # Сдвиг истории
        oldEzRight1[:] = Ez[-3:]            # Текущее → t-1

        # СБОР ДАННЫХ ПРОБ
        for probe in probes:
            probe.addData(Ez, Hy)

        # Обновление анимации каждые 1000 шагов
        if q % 1000 == 0:
            display.updateData(display_field, q)

    # Закрытие анимации
    display.stop()
    tools.showProbeSignals(probes, -1.1, 1.1, dt)  # Графики сигналов проб

    # ==================== СПЕКТРАЛЬНЫЙ АНАЛИЗ ====================
    size = maxTime
    df = 1 / (maxTime * dt)              # Шаг частоты
    start_index_pad = 250 * 10           # Индекс обрезки хвоста падающей волны

    # Заполнение массивов данными проб
    for q in range(1, maxTime):
        Ezspectrumpad[q] = probes[0].E[q]    # Probe 0: падающая волна
        Ezspectrumotr[q] = probes[1].E[q]    # Probe 1: отраженная волна
    Ezspectrumpad[start_index_pad:] = 1e-28  # Обрезка хвоста (избежание wrap-around)

    # Вычисление спектров
    spectrumpad = fft(Ezspectrumpad)
    spectrumotr = fft(Ezspectrumotr)
    koefotr = np.abs(fftshift(spectrumotr / (spectrumpad + 1e-30)))  # Коэффициент отражения Γ(ω)

    # Спектры амплитуд (с центрированием)
    spectrumpad = np.abs(fftshift(fft(Ezspectrumpad)))
    spectrumotr = np.abs(fftshift(fft(Ezspectrumotr)))
    freq = np.arange(-size / 2 * df, size / 2 * df, df)  # Частотная ось
    norm = np.max(spectrumpad)                           # Нормировка

    # ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
    plt.figure(figsize=(12, 5))
    
    # График 1: Спектры падающей и отраженной волн
    plt.subplot(1, 2, 1)
    plt.plot(freq, spectrumpad / norm, label='Падающий')      # Нормированный спектр
    plt.plot(freq, spectrumotr / norm, label='Отражённый')
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax падающего|')
    plt.xlim(0e9, 5e9)      # Диапазон 0-5 ГГц
    plt.legend()

    # График 2: Коэффициент отражения |Γ(ω)|
    plt.subplot(1, 2, 2)
    plt.plot(freq, koefotr)
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|Γ|')
    plt.ylim(0, 0.6)
    plt.xlim(0, 5e9)

    plt.tight_layout()
    plt.show()
