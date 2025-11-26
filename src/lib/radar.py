import numpy as np
from scipy.constants import c

# -------------------------
# Radar Class (with Tx & Rx subclasses)
# -------------------------
class Radar:

    class Transmitter:
        def __init__(self, f0, B, Tc, M, pos):
            self.f0 = f0
            self.B = B
            self.Tc = Tc
            self.M = M
            self.pos = np.array(pos)
            self.k = B / Tc  # chirp slope

        def generate_chirp(self, t):
            return np.exp(1j * 2 * np.pi * (self.f0 * t + 0.5 * self.k * t**2))

    class Receiver:
        def __init__(self, fs, N, pos):
            self.fs = fs
            self.N = N
            self.pos = np.array(pos)

    def __init__(self, transmitter: Transmitter, receiver: Receiver):
        self.tx = transmitter
        self.rx = receiver

# -----------------------------------------------
# Interferer Radar class
# -----------------------------------------------
class InterfererRadar:
    def __init__(self, f0, B, Tc, tx_power=1.0, freq_offset=0, timing_offset=0):
        self.f0 = f0
        self.B = B
        self.Tc = Tc
        self.k = B / Tc
        self.tx_power = tx_power
        self.freq_offset = freq_offset         # Δf between radars
        self.timing_offset = timing_offset     # Δt between chirp start times

    def generate_interference(self, t):
        """
        Generate an interfering FMCW chirp at the victim receiver input.
        """
        # Apply timing offset
        t2 = t - self.timing_offset

        # Interferer's chirp phase
        phase = 2*np.pi * (
            (self.f0 + self.freq_offset) * t2 + 
            0.5 * self.k * (t2**2)
        )

        # Interference amplitude
        sig = np.sqrt(self.tx_power) * np.exp(1j * phase)

        # Interference exists only where t2 >= 0
        sig[t2 < 0] = 0
        return sig