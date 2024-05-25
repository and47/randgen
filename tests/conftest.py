from numpy import random, unique
import pytest


# PARAMS.  As tests grow, can be moved to conftest.py
seed = 123
max_niter = 1_000_000
abs_tol = (1/10**4) * 10  # 10 bps.


# DATA PREP
genor_ran = random.default_rng()  # new per NEP 19, can be used with Cython
genor_fix = random.default_rng(seed)  # to not affect "global" seed


def get_data(genor):
    """In future can be parametrized to return data of provided size, etc."""
    vals = genor.integers(1, high=1//abs_tol, size=2)
    while vals[1] == vals[0]:  # corner case, ensure values are distinct
        vals = genor.integers(1, high=1 // abs_tol, size=2)
    p1 = genor.random()
    probs = (p1, 1 - p1)
    return vals, probs  # tuples for immutability


@pytest.fixture
def userfixed_data() -> tuple:
    vals = [-1, 0, 1.0, 2, 3]  # 1.0 is a manual case to ensure that for int an int is returned, and float for float
    probs = [0.01, 0.3, 0.58, 0.1, 0.01]
    return vals, probs


@pytest.fixture
def rngfixed_data() -> tuple:
    return get_data(genor_fix)  # fixed seed for reproducibility


@pytest.fixture
def rngdyn_data() -> tuple:
    return get_data(genor_ran)


@pytest.fixture
def rngdyn_biggerdata_eqfreq() -> tuple:
    vals = genor_ran.integers(low=-1_000_000, high=1_000_00, size=genor_ran.integers(100, 1000), dtype=int)
    probs = None  # will be made equally likely
    return unique(vals), probs
