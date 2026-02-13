from src.util import fidelity_simulation
import numpy as np


def fidelity_angle(angle: float) -> float:
    """
    Simulate the fidelity of an analog rotation with a given angle.
    For simplicity, we assume the fidelity is a function of the angle, e.g., F(angle) = 1 - (angle / (2 * pi))^2
    """
    return 1 - (angle / (2 * np.pi)) ** 2


logical_fidelity_setting = {"cnot": 1, "rz": fidelity_angle}
physical_fidelity_setting = {"cnot": 1, "rz": 1, "T": 1.5 * 10**6}
