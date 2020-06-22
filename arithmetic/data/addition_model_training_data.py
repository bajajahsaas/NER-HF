#!/usr/bin/env python3

from random import randint, sample, shuffle, uniform
from typing import List, Tuple
import pandas as pd
import numpy as np

def zero_sum_tup(n: int, e: int) -> Tuple[int, ...]:
    """Generate a tuple of n integers,
    (roughly) between -10**e and 10**e,
    which sum to 0.
    """
    L = [randint(-10**e, 10**e) for _ in range(n)]
    mu = sum(L) / n
    L = [int(x - mu) for x in L]
    if sum(L) < 0:
        L = [-x for x in L]
    while sum(L) > 0:
        L[randint(0, n - 1)] -= 1
    return tuple(L)


def wrong_sum_tup(n: int, e: int) -> Tuple[int, ...]:
    """Return a tuple that does not (usually) sum to zero.
    The expected value of the sum of one of these
    is about 0.5 * 10**e (in absolute value).
    """
    return tuple(sample(zero_sum_tup(n + 1, e), n))


def digits(n: int) -> str:
    if n < 0:
        return '-{}'.format(digits(-n))
    elif n == 0:
        return '0'
    elif n < 10:
        return str(n)
    else:
        return '{}{}'.format(digits(n // 10), n % 10)


def inject_errors(n: int) -> int:
    """Inject OCR errors into n."""
    drop_initial_digit_rate = 0.075
    drop_final_digit_rate = 0.075
    change_random_inner_digit_rate = 0.01

    # These numbers result in about half of the tuples
    # having an error in at least one entry.

    s = digits(abs(n))

    while len(s) > 1 and uniform(0, 1) < drop_initial_digit_rate:
        s = s[1:]
    while len(s) > 1 and uniform(0, 1) < drop_final_digit_rate:
        s = s[:-1]

    s = ''.join(str(randint(0, 9)) if uniform(0, 1) < change_random_inner_digit_rate
            else c for c in s)

    return int(s) * (-1 if n < 0 else 1)


def inject_errors_and_label(tup: Tuple[int, ...]) -> Tuple[Tuple[int, ...], bool]:
    """Given a tuple of integers,
    inject some "OCR errors" into the integers and return the modified tuple,
    with a label signifying whether the unmodified tuple summed to zero.
    """
    return (tuple(inject_errors(n) for n in tup), sum(tup) == 0)


def training_samples(num_zero_sum = 10000, num_wrong_sum = 10000) \
        -> List[Tuple[Tuple[int, ...], bool]]:
    """Generate training samples of tuples of integers,
    some of which "sum to zero" and some of which do not.
    """

    L = []

    min_tup_length = 3
    max_tup_length = 5
    min_digit_count = 4
    max_digit_count = 9

    if num_zero_sum > 0:
        L += [zero_sum_tup(randint(min_tup_length, max_tup_length),
            randint(min_digit_count, max_digit_count)) for _ in range(num_zero_sum)]
    if num_wrong_sum > 0:
        L += [wrong_sum_tup(randint(min_tup_length, max_tup_length),
            randint(min_digit_count, max_digit_count)) for _ in range(num_wrong_sum)]

    # L is a list of tuples of integers, some of which actually sum to zero
    # and some of which do not.

    shuffle(L) # I guess this probably isn't necessary?

    return [inject_errors_and_label(tup) for tup in L]

def train_test_split(path, data, per=0.7):
    # TRAIN-TEST SPLIT
    msk = np.random.rand(len(data)) < per
    train = data[msk]
    test = data[~msk]
    train.to_csv(path + '-train.csv', index=False)
    test.to_csv(path + '-test.csv', index=False)

if __name__ == '__main__':
    samples = training_samples()
    labels = []
    strings = []
    for sample in samples:
        tag = sample[1]
        if tag is True:
            label = 1
        else:
            label = 0

        data = sample[0]
        st = ""
        for d in data:
            st = st + str(d) + " "
        strings.append(st)
        labels.append(label)

    data = pd.DataFrame(list(zip(strings, labels)), columns=['context', 'label'])
    path = 'dataset'
    train_test_split(path, data)

    