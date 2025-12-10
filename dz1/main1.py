import numpy as np
import matplotlib.pyplot as plt
from objects import LayerContinuous, LayerDiscrete, Probe
import boundary
import tools

Z0 = 120.0 * np.pi
c0 = 299_792_458.0
eps0 = 8.854187817e-12

Lx = 1.5
eps_r = 4.0
fmin, fmax = 3e9, 8e9
fc = 0.5 * (fmin + fmax)
bw = fmax - fmin

nx_cells = 600
dx = Lx / nx_cells
Sc = 0.99
dt = Sc * dx / c0

class Sampler:
    def __init__(self, d): self.d = float(d)
    def sample(self, x): return int(np.floor(x / self.d + 0.5))

sx = Sampler(dx)
st = Sampler(dt)

maxTime_s = 12e-9
maxTime = st.sample(maxTime_s)
maxSize = sx.sample(Lx)
if maxSize < 3: raise ValueError("Сетка слишком мелкая: maxSize должно быть >= 3")

sourcePos = sx.sample(Lx * 0.01)
probePos = sx.sample(2 * Lx / 3)
probes = [Probe(probePos, maxTime)]

layer_cont = LayerContinuous(0.0, xmax=Lx, eps=eps_r, sigma=0.0)
layer = LayerDiscrete(
    sx.sample(layer_cont.xmin),
    sx.sample(layer_cont.xmax),
    layer_cont.eps,
    1.0,
    layer_cont.sigma
)

eps = np.ones(maxSize) * layer.eps
mu = np.ones(maxSize - 1) * layer.mu
sigma = np.zeros(maxSize)
loss = (sigma * dt) / (2.0 * eps * eps0)
ceze = (1.0 - loss) / (1.0 + loss)
cezh = Z0 / (eps * (1.0 + loss))

Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize - 1)
left_bc = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
right_bc = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

t0 = 3.0 / fmin
tau = 1.0 / bw

def gauss_mod(n):
    t = n * dt
    g = np.exp(-((t - t0) ** 2) / (tau ** 2))
    return g * np.cos(2.0 * np.pi * fc * t)

class SimpleTFSF:
    def __init__(self, pos): self.pos = pos
    def getH(self, n): return -gauss_mod(n + 0.5)
    def getE(self, n): return gauss_mod(n)

source = SimpleTFSF(sourcePos)

speed_refresh = 15
display_field = Ez
display_ylabel = 'Ez, В/м'
display_ymin = -200
display_ymax = 200

display = tools.AnimateFieldDisplay(
    dx, dt, maxSize,
    display_ymin, display_ymax,
    display_ylabel,
    title='fdtd_variant3'
)
display.activate()
display.drawSources([sourcePos])
display.drawProbes([probePos])
display.drawBoundary(layer.xmin)
display.drawBoundary(layer.xmax)

for t in range(1, maxTime):
    Hy += (Ez[1:] - Ez[:-1]) * (Sc / (Z0 * mu))
    if 0 < sourcePos < len(Hy) + 1:
        Hy[sourcePos - 1] += source.getH(t)

    Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy[:-1])
    if 0 <= sourcePos < len(Ez):
        Ez[sourcePos] += source.getE(t)

    left_bc.updateField(Ez, Hy)
    right_bc.updateField(Ez, Hy)

    for p in probes:
        p.addData(Ez, Hy)

    if (t % speed_refresh) == 0:
        display.updateData(display_field, t)

display.stop()

probe = probes[0]
t_axis = np.arange(len(probe.E)) * dt
signal = np.array(probe.E)

plt.rcParams.update({
    "figure.figsize": (9, 4),
    "font.size": 12,
    "lines.linewidth": 1.8,
    "axes.grid": True,
})

plt.figure()
plt.plot(t_axis * 1e9, signal)
plt.xlabel("t, нс")
plt.ylabel("Ez, В/м")
plt.title("Сигнал на датчике")
plt.xlim(0, maxTime_s * 1e9)
plt.tight_layout()

window = np.hanning(len(signal))
spec = np.fft.fft(signal * window)
freq = np.fft.fftfreq(len(signal), dt)
mask_f = freq >= 0
freq_pos = freq[mask_f]
spec_amp = np.abs(spec[mask_f])
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
