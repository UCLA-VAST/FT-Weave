"""Error model for physical operations in the system."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PhysicalErrorModel:
    """Base error model class holding physical error rates.

    The error rates are hard-coded based on physical measurements.
    Provides both basic fix model and computation models with improved rates.
    """

    # Basic model error rates for physical operations
    _BASIC_ERROR_RATES = {
        "coherence_time": 1.5,  # in seconds, for reference
        "init": 3e-3,
        "1q": 3e-3,
        "move": 2.8e-4,  # sum of 4e-5 + 4e-5 + 2e-4
        "move_loss": 2e-5,
        "cz": 2.5e-3,  # CZ gate error: sum of 5e-4 + 6.3e-4 + 1.3e-4
        "cz_spec": 2.3e-3,  # CZ spectator error: sum of 5e-4 + 5e-4 + 1.3e-4
        "cz_loss": 1.3e-3,  # Transfer/movement error
        "measurement": 4e-3,
        "p_ph": 7.3e-3,  # physical error used for compute logical error rates. physical error rate
        # represents the combined error probability for implementing a CZ gate,
        # including transport operations and all associated error channels.
    }

    model_type: str = "basic"  # "basic" or "lookahead"
    lookahead_improvement_ratio: float = 10.0  # Improvement factor for lookahead model

    def __post_init__(self):
        """Validate model type and improvement ratio on initialization."""
        if self.model_type not in ["basic", "lookahead"]:
            raise ValueError(
                f"model_type must be 'basic' or 'lookahead', got {self.model_type}"
            )
        if self.lookahead_improvement_ratio <= 0:
            raise ValueError(
                f"lookahead_improvement_ratio must be positive, got {self.lookahead_improvement_ratio}"
            )

    @property
    def error_rates(self) -> Dict[str, float]:
        """Get error rates for the selected model."""
        if self.model_type == "basic":
            return self._BASIC_ERROR_RATES.copy()
        else:  # lookahead
            return {
                op: rate / self.lookahead_improvement_ratio
                for op, rate in self._BASIC_ERROR_RATES.items()
            }

    def get_error_rate(self, operation: str) -> float:
        """Get the error rate for a specific operation.

        Args:
            operation: Operation name (e.g., 'local_1q', 'cz', 'transfer')

        Returns:
            Error rate as a float

        Raises:
            KeyError: If operation is not recognized
        """
        rates = self.error_rates
        if operation not in rates:
            raise KeyError(
                f"Unknown operation: {operation}. Available: {list(rates.keys())}"
            )
        return rates[operation]

    def get_fidelity(self, operation: str) -> float:
        """Get the fidelity (1 - error_rate) for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Fidelity value (between 0 and 1)
        """
        return 1.0 - self.get_error_rate(operation)

    def switch_model(self, model_type: str) -> None:
        """Switch between basic and lookahead models.

        Args:
            model_type: "basic" or "lookahead"
        """
        if model_type not in ["basic", "lookahead"]:
            raise ValueError(
                f"model_type must be 'basic' or 'lookahead', got {model_type}"
            )
        self.model_type = model_type

    def set_lookahead_improvement(self, ratio: float) -> None:
        """Set the improvement ratio for the lookahead model.

        Args:
            ratio: improvement factor (e.g., 10.0 for 10x improvement)

        Raises:
            ValueError: If ratio is not positive
        """
        if ratio <= 0:
            raise ValueError(f"Improvement ratio must be positive, got {ratio}")
        self.lookahead_improvement_ratio = ratio

    def improvement_factor(self) -> float:
        """Get the improvement factor for the lookahead model.
        Returns:
            Improvement factor (applies uniformly to all operations)
        """
        return self.lookahead_improvement_ratio

    def __repr__(self) -> str:
        """String representation of the error model."""
        rates = self.error_rates
        rates_str = ", ".join(f"{k}: {v:.2e}" for k, v in rates.items())
        return f"ErrorModel(model_type='{self.model_type}', rates={{{rates_str}}})"
