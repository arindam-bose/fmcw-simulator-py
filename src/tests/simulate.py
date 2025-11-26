import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg', 'WxAgg', etc.
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from lib.radar import Radar, InterfererRadar
from lib.target import Target
from lib.process import Process, fmcw_range_fft

# -------------------------
# Instantiate radar and target
# -------------------------
f0 = 77e9
B=1e9
Tc=1e-3
lam = c / f0
tx_positions = [(0, 0), (0, 0.5 * lam)]
rx_positions = [(0, 0.5 * lam), (0, 0)]

tx = Radar.Transmitter(f0=f0, B=B, Tc=Tc, M=2, pos=tx_positions)
rx = Radar.Receiver(fs=2e6, N=2, pos=rx_positions)

radar = Radar(tx, rx)

targets = [Target(pos=(30, 0), v=0, rcs_dbm=-10),
           Target(pos=(60, 0), v=0, rcs_dbm=-20),
           Target(pos=(120, 0), v=0, rcs_dbm=-5)]

# Add interference from another FMCW radar
interferer = InterfererRadar(
        f0=f0,
        B=B,
        Tc=Tc,
        tx_power=0.2,
        freq_offset=2e6,    # 2 MHz carrier offset
        timing_offset=50e-6 # late by 50 µs
)

process = Process(radar, targets, interferer, noise_std=1e-3)

t, dechirped = process.simulate_fmcw()

# single channel example: Tx0-Rx0
sig = dechirped[0, 0, :]

# -------------------------
# Plot time-domain signal
# -------------------------
# plt.figure()
# plt.plot(t, np.real(sig))
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Dechirped Time-Domain Signal (Real Part)")
# plt.show()

# -------------------------
# FFT for Range
# -------------------------
fs = rx.fs
k = tx.k
B = tx.B
Tc = tx.Tc

ranges, mag_db = fmcw_range_fft(sig, fs, k)
rmax = c * (fs / 2) / (2 * B / Tc)

plt.figure()
plt.plot(ranges, mag_db)
plt.vlines(x=rmax, ymin=-20, ymax=60, color='r', linewidth=1, label='Rmax')
plt.vlines(x=-rmax, ymin=-20, ymax=60, color='r', linewidth=1, label='-Rmax')
plt.xlabel("Range (m)")
plt.ylabel("Magnitude (dB)")
plt.title("Range FFT (Tx0-Rx0)")
plt.grid(True)
plt.show()

# coherent sum over all MxN (virtual array coherent sum)
sig_sum = np.sum(dechirped, axis=(0,1))   # sum over tx & rx -> (Ns,)
ranges_sum, mag_db_sum = fmcw_range_fft(sig_sum, fs, k)

plt.figure()
plt.plot(ranges_sum, mag_db_sum)
plt.vlines(x=rmax, ymin=-20, ymax=60, color='r', linewidth=1, label='Rmax')
plt.vlines(x=-rmax, ymin=-20, ymax=60, color='r', linewidth=1, label='-Rmax')
plt.xlabel("Range (m)")
plt.ylabel("Magnitude (dB)")
plt.title("Range FFT (Coherent Sum over MxN)")
plt.grid(True)
plt.show()