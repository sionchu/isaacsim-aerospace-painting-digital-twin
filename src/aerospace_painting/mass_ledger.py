"""Explicit mass-accounting helpers for reference and surrogate runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class MassLedger:
    """Partition the injected mass into reported outcome buckets."""

    injected_kg: float
    deposited_kg: float
    escaped_kg: float
    evaporated_kg: float
    other_kg: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.injected_kg,
            self.deposited_kg,
            self.escaped_kg,
            self.evaporated_kg,
            self.other_kg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("mass ledger values must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("mass ledger values must be non-negative")

    @property
    def accounted_kg(self) -> float:
        return (
            self.deposited_kg
            + self.escaped_kg
            + self.evaporated_kg
            + self.other_kg
        )

    @property
    def balance_error_kg(self) -> float:
        return self.injected_kg - self.accounted_kg

    @property
    def transfer_efficiency(self) -> float:
        if self.injected_kg == 0.0:
            return 0.0
        return self.deposited_kg / self.injected_kg

    def as_dict(self) -> dict[str, float]:
        values = asdict(self)
        values.update(
            accounted_kg=self.accounted_kg,
            balance_error_kg=self.balance_error_kg,
            transfer_efficiency=self.transfer_efficiency,
        )
        return values

    def require_balanced(self, tolerance_kg: float) -> None:
        if tolerance_kg < 0.0 or not np.isfinite(tolerance_kg):
            raise ValueError("tolerance_kg must be finite and non-negative")
        if abs(self.balance_error_kg) > tolerance_kg:
            raise ValueError(
                f"mass ledger does not close: error={self.balance_error_kg:.6g} kg, "
                f"tolerance={tolerance_kg:.6g} kg"
            )
