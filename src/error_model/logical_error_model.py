"""Logical error model for quantum error correction codes."""

from dataclasses import dataclass, field
from typing import Dict
from src.error_model import PhysicalErrorModel
import math


@dataclass
class LogicalErrorModel:
    """Logical error model for surface code error correction.

    Relates physical error rates to logical error rates through code distance.
    Uses the formula: p_logical = prefactor * p_physical^((d+1)/2)
    """

    # Default logical error rates at code distance 7 with p_ph = 7.3e-4
    _DEFAULT_LOGICAL_ERROR_RATES = {
        "I": 2e-7,  # Identity error
        "H": 4e-7,  # Hadamard error
        "S": 1.46e-6,  # S gate error
        "CZ": 1.8e-6,  # CZ gate error
    }

    # Reference parameters for default rates
    _REFERENCE_CODE_DISTANCE = 7
    _REFERENCE_P_PHYSICAL = 7.3e-4

    _THRESHOLD_ERROR_RATE_P_C = 8.5e-3
    _PREFACTOR_A = 0.0579

    # Physical error model instance
    physical_model: PhysicalErrorModel = field(
        default_factory=lambda: PhysicalErrorModel(
            model_type="lookahead", lookahead_improvement_ratio=10.0
        )
    )

    # Code distance for surface code
    code_distance: int = 7

    # Optional logical fidelity for T (T-cultivation); set via
    # ``integrate_t_cultivation_fidelity_target``. When None, ``get_logical_fidelity("T")``
    # falls back to ``get_rotation_fidelity(pi/4)``.
    _t_gate_logical_fidelity: float | None = field(default=None, repr=False)

    # Logical error rates at current configuration
    logical_error_rates = {
        "I": 2e-7,  # Identity error
        "H": 4e-7,  # Hadamard error
        "S": 1.46e-6,  # S gate error
        "CNOT": 1.8e-6
        + 2 * 4e-7
        + 2 * 2e-7,  # CZ error + 2 H errors + 2 identity errors
    }

    def __post_init__(self):
        """Initialize and validate the logical error model."""
        if self.code_distance < 3:
            raise ValueError(f"Code distance must be >= 3, got {self.code_distance}")
        if self.code_distance != 7:
            self._compute_logical_error_rates()

    def _compute_logical_error_rates(self):
        """Compute CNOT error rates based on code distance and physical error rate.

        Uses the formula: p_logical = reference_rate * (p_physical / p_ref)^((d_ref+1)/2) * ((d+1)/2) / ((d_ref+1)/2)

        This scales the reference rates proportionally with physical error rate and code distance.
        """
        # ! only update CNOT error rate for now
        # Scaling factor based on code distance
        p_physical = self.physical_model.get_error_rate("p_ph")
        d_exponent = math.ceil((self.code_distance + 1) / 2)

        # Scale based on physical error rate change
        old_cnot_fid = self.logical_error_rates["CNOT"]
        new_cnot_fid = (
            self._PREFACTOR_A
            * (p_physical / self._THRESHOLD_ERROR_RATE_P_C) ** d_exponent
        )
        error_ratio = new_cnot_fid / old_cnot_fid
        # # Apply scaling to reference rates
        self.logical_error_rates = {
            op: rate * error_ratio for op, rate in self.logical_error_rates.items()
        }

    def set_code_distance(self, distance: int) -> None:
        """Set the code distance and recompute logical error rates.

        Args:
            distance: Code distance (must be odd and >= 3)

        Raises:
            ValueError: If distance is invalid
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Code distance must be odd and >= 3, got {distance}")
        self.code_distance = distance
        self._compute_logical_error_rates()

    def set_physical_error_model(self, physical_model: PhysicalErrorModel) -> None:
        """Set the physical error rate and recompute logical error rates."""
        self.physical_model = physical_model
        self._compute_logical_error_rates()

    def integrate_t_cultivation_fidelity_target(
        self,
        fidelity_target: float,
        *,
        target_is_logical_error_rate: bool = True,
    ) -> None:
        """Set logical T-gate fidelity from T-cultivation ``TSetting.fidelity_target``.

        By default, ``fidelity_target`` is treated as a logical **error rate** ``p``
        (e.g. ``1e-8``); logical T fidelity is ``1 - p``. Set
        ``target_is_logical_error_rate=False`` if ``fidelity_target`` is already a
        fidelity in ``(0, 1]`` near 1.

        Args:
            fidelity_target: Value from evaluation / throughput settings.
            target_is_logical_error_rate: If True (default), store ``1 - fidelity_target``.
        """
        ft = float(fidelity_target)
        if target_is_logical_error_rate:
            self._t_gate_logical_fidelity = 1.0 - ft
        else:
            self._t_gate_logical_fidelity = ft

    def clear_t_gate_logical_fidelity(self) -> None:
        """Unset integrated T fidelity; ``get_logical_fidelity('T')`` uses rotation model."""
        self._t_gate_logical_fidelity = None

    def get_logical_error_rate(self, operation: str) -> float:
        """Get the logical error rate for a specific operation.

        Args:
            operation: Operation name ('I', 'H', 'S', 'CZ', 'CNOT', 'T', ...)

        Returns:
            Logical error rate

        Raises:
            KeyError: If operation is not recognized
        """
        if operation == "T":
            return 1.0 - self.get_logical_fidelity("T")
        if operation not in self.logical_error_rates:
            raise KeyError(
                f"Unknown operation: {operation}. Available: {list(self.logical_error_rates.keys())}"
            )
        return self.logical_error_rates[operation]

    def get_logical_fidelity(self, operation: str) -> float:
        """Get the logical fidelity (1 - error_rate) for a specific operation.

        Args:
            operation: Operation name. For ``'T'``, uses integrated T-cultivation
                target if set, else ``get_rotation_fidelity(pi/4)``.

        Returns:
            Logical fidelity value (between 0 and 1)
        """
        if operation == "T":
            if self._t_gate_logical_fidelity is not None:
                return self._t_gate_logical_fidelity
            return self.get_rotation_fidelity(math.pi / 4)
        return 1.0 - self.get_logical_error_rate(operation)

    def get_rotation_fidelity(self, angle: float) -> float:
        """Get the logical fidelity for a rotation gate based on its angle.

        Assumes the error scales with the angle of rotation.

        Args:
            angle: Rotation angle in radians

        Returns:
            Logical fidelity for the rotation gate
        """
        # For simplicity, we assume the error scales linearly with the angle
        if self.code_distance == 3:
            a = 8.038298e-04
            b = 1.453911
            c = 0.000013
        elif self.code_distance == 5:
            a = 3.827042e-04
            b = 1.463395
            c = 0.000000
        elif self.code_distance == 7 or self.code_distance == 9:
            a = 3.136180e-04
            b = 1.464970
            c = 0.000000
        else:
            raise ValueError(f"Unsupported code distance: {self.code_distance}")
        angle = abs(angle)  # !
        return 1 - a * angle**b + c

    def get_summary(self) -> Dict:
        """Get a summary of the current logical error model configuration.

        Returns:
            Dictionary with model configuration and error rates
        """
        return {
            "code_distance": self.code_distance,
            "p_physical": self.physical_model.get_error_rate("p_ph"),
            "physical_model_type": self.physical_model.model_type,
            "logical_error_rates": self.logical_error_rates.copy(),
        }

    def __repr__(self) -> str:
        """String representation of the logical error model."""
        rates_str = ", ".join(
            f"{k}: {v:.2e}" for k, v in self.logical_error_rates.items()
        )
        p_physical = self.physical_model.get_error_rate("p_ph")
        return f"LogicalErrorModel(d={self.code_distance}, p_ph={p_physical:.2e}, rates={{{rates_str}}})"
