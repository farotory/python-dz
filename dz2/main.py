# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt

# =====================================================
# КЛАСС ИСТОЧНИКА - ГАУССОВ ИМПУЛЬС (БЕЗ МОДУЛЯЦИИ)
# =====================================================

class GaussianPlaneWave:
    """Немодулированный гауссов импульс, как в tr.py"""

    def __init__(self, dt, d, w, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d          # задержка в секундах
        self.dt = dt        # шаг по времени
        self.w = w          # ширина импульса в секундах
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        # q и m — как в TF/SF: q — индекс времени, m — индекс ячейки
        return np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc)
                          - self.d / self.dt) / (self.w / self.dt)) ** 2)

# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def create_layer_boundaries(layer_x_start, d1, d2, d3, d):
    """Перевод физических координат границ слоев в индексы дискретной сетки"""
    layer1_x = layer_x_start        # Начало 1-го слоя
    layer2_x = layer1_x + d1        # Граница 1-2 слоя
    layer3_x = layer2_x + d2        # Граница 2-3 слоя
    layer4_x = layer3_x + d3        # Граница 3-4 слоя

    layer1_DX = int(np.ceil(layer1_x / d))
    layer2_DX = int(np.ceil(layer2_x / d))
    layer3_DX = int(np.ceil(layer3_x / d))
    layer4_DX = int(np.ceil(layer4_x / d))

    return [layer1_DX, layer2_DX, layer3_DX, layer4_DX]

def setup_abc_coefficients(Sc, mu, eps):
    """Вычисление коэффициентов ABC (Mur 2nd order) для поглощающих границ"""
    # ЛЕВАЯ ГРАНИЦА (i=0)
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])
    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # ПРАВАЯ ГРАНИЦА (i=maxSize-1)
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    return (k1Left, k2Left, k3Left, k4Left,
            k1Right, k2Right, k3Right, k4Right)

# =====================================================
# ОСНОВНАЯ ПРОГРАММА - 1D FDTD СИМУЛЯЦИЯ
# =====================================================

if __name__ == '__main__':

    # ==================== ПАРАМЕТРЫ ДИСКРЕТИЗАЦИИ ====================
    d = 0.3e-3                      # Шаг пространственной сетки, м (0.3 мм)
    Z0 = 120.0 * np.pi              # Характеристическое сопротивление
    Sc = 1.0                        # Коэффициент Кори
    mu0 = np.pi * 4e-7              # Магн. прониц. вакуума
    eps0 = 8.854187817e-12          # Электр. прониц. вакуума
    c = 1.0 / np.sqrt(mu0 * eps0)   # Скорость света
    dt = d / c * Sc                 # Шаг по времени

    print(f"dt = {dt}")

    maxTime_sec = 25e-9
    maxTime = int(np.ceil(maxTime_sec / dt))

    sizeX_m = 1.0
    maxSize = int(np.ceil(sizeX_m / d))

    # ==================== ПАРАМЕТРЫ СЛОЕВ ====================
    d1, d2, d3 = 0.06, 0.06, 0.10
    eps1, eps2, eps3, eps4 = 3.5, 2.2, 2.0, 6.0

    layer_x_start = 0.52
    layer_boundaries = create_layer_boundaries(layer_x_start, d1, d2, d3, d)
    layer1_DX, layer2_DX, layer3_DX, layer4_DX = layer_boundaries

    # ==================== ИСТОЧНИК И ПРОБЫ ====================
    sourcePosm = 0.1
    sourcePos = int(np.ceil(sourcePosm / d))

    probesPos = [int(np.ceil(0.1 / d)), int(np.ceil(0.01 / d))]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # ==================== ПРОФИЛЬ СРЕДЫ ====================
    eps = np.ones(maxSize)
    eps[layer1_DX:] = eps1
    eps[layer2_DX:] = eps2
    eps[layer3_DX:] = eps3
    eps[layer4_DX:] = eps4

    mu = np.ones(maxSize - 1)


    # ==================== ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ====================
    Ez = np.zeros(maxSize)
    Ezspectrumpad = np.zeros(maxTime)
    Ezspectrumotr = np.zeros(maxTime)
    Hy = np.zeros(maxSize - 1)

    # ==================== ГАУССОВ ИМПУЛЬС (БЕЗ МОДУЛЯЦИИ) ====================
    # Аналог параметризации из tr.py: A_0, A_max, F_max -> w_g, d_g
    A_0 = 100.0        # стартовое отношение экспоненты
    A_max = 100.0      # макс. затухание
    F_max = 5e9        # характерная частота (как раньше 5 ГГц)
    w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)
    d_g = w_g * np.sqrt(np.log(A_0))*2

    print(f"w_g = {w_g}")
    print(f"d_g = {d_g}")

    source = GaussianPlaneWave(dt, d_g, w_g, Sc, eps[sourcePos], mu[sourcePos])

    # ==================== ABC КОЭФФИЦИЕНТЫ ====================
    (k1Left, k2Left, k3Left, k4Left,
     k1Right, k2Right, k3Right, k4Right) = setup_abc_coefficients(Sc, mu, eps)

    oldEzLeft1 = np.zeros(3)
    oldEzLeft2 = np.zeros(3)
    oldEzRight1 = np.zeros(3)
    oldEzRight2 = np.zeros(3)

    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin, display_ymax = -1.1, 1.1

    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, d, dt)
    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(layer1_DX)
    display.drawBoundary(layer2_DX)
    display.drawBoundary(layer3_DX)
    display.drawBoundary(layer4_DX)

    # ==================== ОСНОВНОЙ ЦИКЛ FDTD (Yee scheme) ====================
    for q in range(1, maxTime):
        # H-обновление
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        # E-обновление
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        # Источник (hard source) в E, со сдвигом на полшага
        Ez[sourcePos] += (Sc / np.sqrt(eps[sourcePos] * mu[sourcePos]) *
                          source.getE(-0.5, q + 0.5))

        # ABC слева
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2]
                                     - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])
        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0:3]

        # ABC справа
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3]
                                        - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])
        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Сбор данных проб
        for probe in probes:
            probe.addData(Ez, Hy)

        # Обновление анимации
        if q % 1000 == 0:
            display.updateData(display_field, q)

    display.stop()
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # ==================== СПЕКТРАЛЬНЫЙ АНАЛИЗ ====================
    size = maxTime
    df = 1 / (maxTime * dt)
    start_index_pad = 250 * 10

    for q in range(1, maxTime):
        Ezspectrumpad[q] = probes[0].E[q]
        Ezspectrumotr[q] = probes[1].E[q]

    Ezspectrumpad[start_index_pad:] = 1e-28

    spectrumpad = fft(Ezspectrumpad)
    spectrumotr = fft(Ezspectrumotr)
    koefotr = np.abs(fftshift(spectrumotr / (spectrumpad + 1e-30)))

    spectrumpad = np.abs(fftshift(fft(Ezspectrumpad)))
    spectrumotr = np.abs(fftshift(fft(Ezspectrumotr)))

    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    norm = np.max(spectrumpad)

    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(freq, spectrumpad / norm, label='Падающий')
    plt.plot(freq, spectrumotr / norm, label='Отражённый')
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax падающего|')
    plt.xlim(0e9, 5e9)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(freq, koefotr)
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|Γ|')
    plt.ylim(0, 0.6)
    plt.xlim(0, 5e9)
    plt.tight_layout()
    plt.show()
