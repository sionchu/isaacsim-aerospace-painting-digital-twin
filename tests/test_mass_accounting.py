import pytest

from aerospace_painting.mass_ledger import MassLedger


def test_mass_ledger_closes_and_reports_transfer_efficiency():
    ledger = MassLedger(
        injected_kg=1.0,
        deposited_kg=0.6,
        escaped_kg=0.25,
        evaporated_kg=0.1,
        other_kg=0.05,
    )
    ledger.require_balanced(1e-12)
    assert ledger.balance_error_kg == 0.0
    assert ledger.transfer_efficiency == 0.6


def test_mass_ledger_rejects_unaccounted_mass():
    ledger = MassLedger(
        injected_kg=1.0,
        deposited_kg=0.4,
        escaped_kg=0.2,
        evaporated_kg=0.1,
    )
    with pytest.raises(ValueError, match="does not close"):
        ledger.require_balanced(1e-12)
