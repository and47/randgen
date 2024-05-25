import random
from numpy import cumsum, searchsorted, isclose
from typing import List, Protocol, Iterator

class Rng(Protocol):
    def random(self) -> float:
        pass

    def seed(self, s: int) -> None:
        pass


class RandomGen:
    def __init__(self, random_nums: List[int | float], probabilities: List[float] | None = None,
                 seed: int | None = None, rng: Rng | None = None):
        """
        Creates a random number generator, which will output specified numbers with given probabilities.
        Is not lock-safe (see e.g. `RandomGen.remove`).
        :param random_nums: input numbers to generate (choose) from. can be Iterable, tested with List
        :param probabilities: respective (aligned) to numbers; define likelihood of seeing each number in the output;
            default to equal probabilities for each `random_nums`
        :param seed: (optional) RNG seed, see e.g. `help(random.seed)`
        :param rng: (optional) use your choice of RNG to generate [0, 1), internally. should implement .seed and .random,
         see `Rng`, e.g. numpy's random (rng=np.random.default_rng()). defaults to `random.Random`.

        :examples: see `test_custom_rngs.py`
        """
        if rng is None:
            self.rng = random.Random()  # local instead of global instance to not affect outside user env
        else:
            self.rng = rng
        self._seed = seed
        if seed is not None:
            self.rng.seed(seed)
        self.nums = random_nums
        n = len(self.nums)
        if n == 0:
            raise ValueError("At least one random number value is required")
        if len(set(self.nums)) < n:
            raise ValueError("Values cannot repeat, ambiguous")
        if probabilities:
            if len(probabilities) != len(random_nums): raise ValueError("Inputs not compatible")
            if ~isclose(sum(probabilities), 1) or any(p <= 0 or p > 1 for p in probabilities):
                raise ValueError("Invalid probability")
            self.probs = probabilities
        else:
            self.probs = [1 / n] * n
        self.cdf = cumsum(self.probs)  # pre-compute the cumulative distribution function

    @classmethod
    def from_dict(cls, nums_probs: dict, seed: int | None = None, rng: Rng | None = None) -> "RandomGen":
        """Initialize an instance from a dictionary mapping numbers (keys) to probabilities (values)."""
        return cls(random_nums=list(nums_probs.keys()), probabilities=list(nums_probs.values()), seed=seed, rng=rng)

    def __repr__(self) -> str:  # should be preset, and updated when state changes, left for short
        return str(dict(sorted(zip(self.nums, [f"{p=:.2%}" for p in self.probs]), key=lambda x: x[1], reverse=True)))

    def __getitem__(self, item: int | float) -> float:
        """Get probability for a given number (value)"""
        return dict(zip(self.nums, self.probs))[item]

    def __contains__(self, item) -> bool:
        return item in self.nums

    def __iter__(self):
        return self  # could be made unnecessary if inheriting from collections.abc.Iterator: RandomGen(Iterator)

    def __next__(self) -> int | float:
        """
        Returns one of the randomNums. When this method is called multiple
        times over a long period, it should return the numbers roughly with
        the initialized probabilities.
        """
        rnum = self.rng.random()  # generate a random float in [0, 1)
        idx = searchsorted(self.cdf, rnum)  # find the index where this random number would fit in the CDF
        return self.nums[idx]  # return the corresponding random number

    def next_num(self) -> int | float:
        return next(self)

    def next_k_nums(self, k: int) -> Iterator:
        """Example use as generator. For inf k, iterate over instance itself."""
        if isinstance(k, int) and k > 0:
            while k:
                k -= 1
                yield next(self)
        else:
            raise ValueError("Invalid k: must be a positive integer")

    # below are only example interfaces, a complete implementation could similarly allow to update values,
    #  probabilities, and to prevent self.cdf from being changed directly.  also, can be done using dataclasses
    def remove(self, *, value: int | float | None = None, index: int | None = None) -> None:
        """Remove element (random number) by its value or index, equally affecting remaining elements' probabilities
        :param index: mostly for future functionality, e.g. if possible to add new numbers and probabilities, this can
         be used to remove e.g. last added one
        """
        if not ((value is None) ^ (index is None)):
            raise ValueError("To remove element, specify either its value or its index")
        n_prev = len(self.nums)
        if value is None:
            if not -n_prev <= index < n_prev:
                raise IndexError(f"Index {index} is out of the valid range for length {n_prev - 1}.")
            value = self.nums[index]
        if index is None:
            if value not in self:
                raise "Passed value is not present in RNG's possible numbers"
            index = self.nums.index(value)
        if n_prev <= 1:
            raise ValueError("Cannot remove only one number left. Instead, create a new RandomGen instance without it")
        delta = self.probs.pop(index) / (n_prev - 1)
        self.nums.remove(value)
        self.probs = [p + delta for p in self.probs]  # equally update probabilities of remaining numbers
        self.cdf = cumsum(self.probs)  # pre-compute the cumulative distribution function

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter  # by convention this is often done as callable instead;  for simplicity, left as is
    def seed(self, new_seed: int) -> None:
        self.rng.seed(new_seed)
        self._seed = new_seed
