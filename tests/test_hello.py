from RCAEval.e2e.baro import baro

def test_basic():
    assert baro() is not None