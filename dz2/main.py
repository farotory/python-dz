import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt


class GaussianDiff:
    def __init__(self, dt, A, F, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.A = A
        self.F = F
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.dt = dt

    def getE(self, m, q):
        w = 2 * np.sqrt(np.log(self.A)) / (np.pi * self.F)
        d = w * np.sqrt(np.log(self.A))
        dt = self.dt
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - d / dt) / (w / dt)) ** 2))


if __name__ == '__main__':

    d = 0.35e-3
    Z0 = 120.0 * np.pi
    Sc = 1.0
    mu0 = np.pi * 4e-7
    eps0 = 8.854187817e-12
    c = 1.0 / np.sqrt(mu0 * eps0)

    dt = d / c * Sc
    print(dt)
    maxTime_sec = 8e-9
    maxTime = int(np.ceil(maxTime_sec / dt))

    sizeX_m = 1
    maxSize = int(np.ceil(sizeX_m / d))

    d1 = 0.06
    d2 = 0.06
    d3 = 0.10

    eps1 = 3.5
    eps2 = 2.2
    eps3 = 2.0
    eps4 = 6.0

    layer_x_start = 0.52

    layer1_x = layer_x_start
    layer2_x = layer1_x + d1
    layer3_x = layer2_x + d2
    layer4_x = layer3_x + d3

    layer1_DX = int(np.ceil(layer1_x / d))
    layer2_DX = int(np.ceil(layer2_x / d))
    layer3_DX = int(np.ceil(layer3_x / d))
    layer4_DX = int(np.ceil(layer4_x / d))

    sourcePosm = 0.3
    sourcePos = int(np.ceil(sourcePosm / d))

    probesPos = [int(np.ceil(0.4 / d)), int(np.ceil(0.22 / d))]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    eps = np.ones(maxSize)
    eps[layer1_DX:] = eps1
    eps[layer2_DX:] = eps2
    eps[layer3_DX:] = eps3
    eps[layer4_DX:] = eps4
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Ezspectrumpad = np.zeros(maxTime)
    Ezspectrumotr = np.zeros(maxTime)
    Hy = np.zeros(maxSize - 1)

    source = GaussianDiff(dt,40, 2e9, 445)

    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    oldEzLeft1 = np.zeros(3)
    oldEzLeft2 = np.zeros(3)
    oldEzRight1 = np.zeros(3)
    oldEzRight2 = np.zeros(3)

    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

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

    for q in range(1, maxTime):
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        Sc1Left = Sc / np.sqrt(mu[0] * eps[0])
        k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
        k2Left = 1 / Sc1Left - 2 + Sc1Left
        k3Left = 2 * (Sc1Left - 1 / Sc1Left)
        k4Left = 4 * (1 / Sc1Left + Sc1Left)

        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0:3]

        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 1000 == 0:
            display.updateData(display_field, q)

    display.stop()

    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    size = maxTime
    df = 1 / (maxTime * dt)

    Ezspectrumpad[:] = 0.0
    Ezspectrumotr[:] = 0.0

    start_index_pad = 250 * 10

    for q in range(1, maxTime):
        Ezspectrumpad[q] = probes[0].E[q]
        Ezspectrumotr[q] = probes[1].E[q]

    Ezspectrumpad[start_index_pad:] = 1e-28

    spectrumpad = fft(Ezspectrumpad)
    spectrumotr = fft(Ezspectrumotr)
    koefotr = spectrumotr / (spectrumpad + 1e-30)
    koefotr = np.abs(koefotr)
    koefotr = fftshift(koefotr)

    spectrumpad = np.abs(fft(Ezspectrumpad))
    spectrumotr = np.abs(fft(Ezspectrumotr))

    spectrumotr = fftshift(spectrumotr)
    spectrumpad = fftshift(spectrumpad)

    freq = np.arange(-size / 2 * df, size / 2 * df, df)

    norm = np.max(spectrumpad)

    plt.subplot(1, 1, 1)
    plt.plot(freq, spectrumpad / norm, label='Падающий')
    plt.plot(freq, spectrumotr / norm, label='Отражённый')
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax падающего|')
    plt.xlim(0e9, 5e9)
    plt.legend()


    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(freq, koefotr)
    plt.grid()
    plt.ylabel('|Г|')
    plt.ylim(0, 0.6)
    plt.xlim(0, 5e9)

    plt.subplots_adjust(wspace=0.4)
    plt.show()
