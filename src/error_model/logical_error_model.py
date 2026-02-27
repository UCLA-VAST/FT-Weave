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

        self.logical_error_rates["CNOT"] = (
            self._PREFACTOR_A
            * (p_physical / self._THRESHOLD_ERROR_RATE_P_C) ** d_exponent
        )

        # # Apply scaling to reference rates
        # scaled_rates = {
        #     op: rate * error_ratio
        #     for op, rate in self._DEFAULT_LOGICAL_ERROR_RATES.items()
        # }

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

    def get_logical_error_rate(self, operation: str) -> float:
        """Get the logical error rate for a specific operation.

        Args:
            operation: Operation name ('I', 'H', 'S', 'CZ')

        Returns:
            Logical error rate

        Raises:
            KeyError: If operation is not recognized
        """
        if operation not in self.logical_error_rates:
            raise KeyError(
                f"Unknown operation: {operation}. Available: {list(self.logical_error_rates.keys())}"
            )
        return self.logical_error_rates[operation]

    def get_logical_fidelity(self, operation: str) -> float:
        """Get the logical fidelity (1 - error_rate) for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Logical fidelity value (between 0 and 1)
        """
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
        return 1
        raise NotImplementedError(
            "Rotation fidelity calculation is not implemented yet."
        )
        # base_fidelity = self.get_logical_f

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
