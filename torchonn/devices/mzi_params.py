class MZIParams:
    """Parámetros físicos del interferómetro Mach-Zehnder."""
    def __init__(self, wavelength: float = 1550e-9, loss_dB: float = 0.1):
        self.wavelength = wavelength
        self.loss_dB = loss_dB