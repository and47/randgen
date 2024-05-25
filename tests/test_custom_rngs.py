import pytest
from numpy import array, isclose
from numbers import Integral, Real  # these are abstract types, used e.g. so that both numpy and Python ints are ok
from custom_rngs import RandomGen
from conftest import genor_fix, genor_ran, abs_tol, max_niter  # unlike fixtures, vars & funs aren't available automatically


def init_mygen(*args, **kwargs) -> RandomGen:
    return RandomGen(*args, **kwargs)


# TESTS
@pytest.mark.parametrize('datafixture', ['userfixed_data', 'rngfixed_data', 'rngdyn_data', 'rngdyn_biggerdata_eqfreq'])
def test_next_num(datafixture, request):
    data = request.getfixturevalue(datafixture)
    vals, _ = data
    myrng = init_mygen(*data)

    for _ in range(42):  # to not test only once
        outval = myrng.next_num()
        outtype = type(outval)
        assert isinstance(outval, Integral) or isinstance(outval, Real), \
            f"Invalid value type returned, expected a single value int or float, got {outtype}"
        assert outval in vals, "Unexpected value returned"
        assert outtype == type(vals[list(vals).index(outval)]), "Invalid type returned"  # list.index method not in ndarray


@pytest.mark.parametrize('datafixture', ['userfixed_data', 'rngfixed_data', 'rngdyn_data', 'rngdyn_biggerdata_eqfreq'])
def test_convergence(datafixture, request):
    # Checks that output frequencies (ratios) are valid (converge to prior probabilities)
    data = request.getfixturevalue(datafixture)
    vals, probs = data
    if probs is None:
        probs = 1 / len(vals)
    myrng = init_mygen(*data)

    valuecounts = {k: 0 for k in vals}  # initiate 0s as a prob could be tiny, and lengths of ratios & probs won't match
    for i, v in enumerate(myrng):  # infinite iterator
        valuecounts[v] += 1
        counts = array(list(valuecounts.values()))
        ratios = counts / sum(counts)
        if all(isclose(ratios, probs, atol=abs_tol)):
            break
        if i > max_niter:
            raise AssertionError(f"Reached max N iterations set ({max_niter}), and failed to converge at {abs_tol}: "
                                 f"{max(abs(ratios - probs))}")


@pytest.mark.parametrize('datafixture', ['userfixed_data', 'rngfixed_data', 'rngdyn_data', 'rngdyn_biggerdata_eqfreq'])
def test_spec(datafixture, request):
    # quick test of valid specification, produces expected (set) values of requested number
    data = request.getfixturevalue(datafixture)
    myrng = init_mygen(*data)
    # requested lens of output
    ks = 42, genor_fix.integers(0, 1//abs_tol), genor_ran.integers(0, 1//abs_tol)

    for k in ks:
        res = []
        mygen = myrng.next_k_nums(k)  # re-use one of values as k
        for v in mygen:
            res.append(v)
        assert len(res) == k, f"Unexpected output length"
        assert set(res).issubset(data[0]), \
                        f"Unexpected values encountered from generator {set(data[0]) - set(res)}"


@pytest.mark.parametrize("vals, probs, expected_exception", [
    ([-1, 0, 1], [0.3, 0.3, 0.4], None),
    ([], [], ValueError),
    ([-1, 0, 1], [0.1, 0.1], ValueError),
    ([-1, 0, 1], [-0.1, 1.1, 0.0], ValueError)
])
def test_initialization(vals, probs, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            init_mygen(vals, probs)
    else:
        rng = init_mygen(vals, probs)
        assert isinstance(rng, RandomGen), "Failed to initialize RandomGen properly."
