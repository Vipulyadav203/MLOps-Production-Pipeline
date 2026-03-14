import pytest
from pydantic import ValidationError
from app.inference import Transaction

def test_transaction_validation_success():
    """
    Test valid transaction input.
    """
    data = {
        "amount": 100.50,
        "time": 3600.0,
        "v1": 0.5,
        "v2": -0.3
    }
    tx = Transaction(**data)
    assert tx.amount == 100.50
    assert tx.time == 3600.0

def test_transaction_validation_failure():
    """
    Test invalid transaction input (amount as string).
    """
    data = {
        "amount": "not-a-number",
        "time": 3600.0,
        "v1": 0.5,
        "v2": -0.3
    }
    with pytest.raises(ValidationError):
        Transaction(**data)

def test_missing_field():
    """
    Test missing field in transaction input.
    """
    data = {
        "amount": 100.50,
        "time": 3600.0,
        "v1": 0.5
        # Missing v2
    }
    with pytest.raises(ValidationError):
        Transaction(**data)
