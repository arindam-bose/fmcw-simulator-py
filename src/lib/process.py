import numpy as np
from typing import List
from scipy.constants import c

from lib.radar import Radar, InterfererRadar
from lib.target import Target

class Process:
    def __init__(self, radar: Radar, targets: List[Target], 
                 interferer: InterfererRadar, noise_std=1e-3):
        self.radar = radar
        self.targets = targets
        self.interferer = interferer
        self.noise_std = noise_std

    def simulate_fmcw(self):
        fs = self.radar.rx.fs
        Tc = self.radar.tx.Tc
        Ns = int(fs * Tc)
        t = np.arange(Ns) / fs
        k = self.radar.tx.k
        f0 = self.radar.tx.f0
        lam = c / f0

        tx_pos = np.array(self.radar.tx.tx_positions)   # shape (M, 2)
        rx_pos = np.array(self.radar.rx.rx_positions)   # shape (N, 2)

        # Transmit chirp
        tx_chirp = self.radar.tx.generate_chirp(t)

        # Received signal buffer
        rx_sig = np.zeros((self.radar.tx.M, self.radar.rx.N, Ns), dtype=complex)

        # Loop through targets
        for tgt in self.targets:
            p = tgt.pos  # (x, y)
            scale = 10**(tgt.rcs_dbm / 20)

            # Doppler frequency = 2*v / λ
            fd = 2 * tgt.v / lam

            for m in range(self.radar.tx.M):
                for n in range(self.radar.rx.N):
                    # Geometric distances
                    d_tx = np.linalg.norm(tx_pos[m] - p)
                    d_rx = np.linalg.norm(p - rx_pos[n])
                    R = d_tx + d_rx

                    tau = R / c

                    # Target return for this TX-RX pair
                    delayed = np.exp(
                        1j * 2 * np.pi * (
                            f0 * (t - tau) + 0.5 * k * (t - tau)**2 + fd * t
                        )
                    )
                    
                    # Accumulate into receive channel
                    rx_sig[m, n, :] += scale * delayed

        # Add Interference
        if self.interferer is not None:
            interference = self.interferer.generate_interference(t)
            rx_sig += interference[None, None, :]

        # Add noise
        rx_sig += (np.random.randn(*rx_sig.shape) + 1j*np.random.randn(*rx_sig.shape)) * self.noise_std

        # Dechirp (mix with conjugate of TX chirp)
        dechirped = rx_sig * np.conj(tx_chirp)[None, None, :]

        return t, dechirped
    
def fmcw_range_fft(rx_signal, fs, k):
    """
    Compute the FMCW range FFT (positive frequencies only).

    Parameters
    ----------
    rx_signal : ndarray (Ns,)
        The dechirped time-domain signal.
    fs : float
        Sampling rate in Hz.
    k : float
        Chirp slope in Hz/s (B / T_chirp).

    Returns
    -------
    ranges : ndarray
        Range axis (meters), positive only.
    S_db : ndarray
        Magnitude spectrum (dB) corresponding to ranges.
    """

    Ns = len(rx_signal)

    # FFT + shift
    S = np.fft.fft(rx_signal)
    S = np.fft.fftshift(S)

    # Magnitude in dB
    S_db = 20 * np.log10(np.abs(S) + 1e-12)

    # Frequency axis (shifted)
    fb = np.fft.fftfreq(Ns, d=1/fs)
    fb = np.fft.fftshift(fb)

    # Convert beat frequency --> range
    ranges = c * fb / (2 * k)

    # Keep only physical positive ranges
    # keep = ranges >= 0
    return ranges, S_db
