import numpy as np

# -------------------------
# Target Class
# -------------------------
class Target:
    def __init__(self, pos, v, rcs_dbm):
        self.pos = np.array(pos, dtype=float)    # (x, y) position
        self.v = v              # velocity (m/s)
        self.rcs_dbm = rcs_dbm  # intensity (dBm)