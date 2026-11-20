"""Exact coefficient check for BBFM Conjecture 4, using only Python integers."""

import argparse


def check(limit):
    coeffs = [1]
    count = 0
    failures = {}
    for n in range(1, limit + 1):
        # The exponent of (1+x^j) increases precisely when n/j is a power of two.
        j = n
        while True:
            old_degree = len(coeffs) - 1
            coeffs.extend([0] * j)
            for i in range(old_degree, -1, -1):
                coeffs[i + j] += coeffs[i]
            if j % 2:
                break
            j //= 2
        degree = len(coeffs) - 1
        assert degree == sum(j * (n // j).bit_length() for j in range(1, n + 1))
        assert coeffs == coeffs[::-1]
        assert sum(coeffs) == 2 ** sum((n // j).bit_length() for j in range(1, n + 1))
        for i in range(1, degree // 2 + 1):
            a, b, c = coeffs[i - 1:i + 2]
            if b * b < a * c and n not in failures:
                failures[n] = (i, a, b, c)
            count += 1
    print(f"n = 1..{limit}, comparisons up to symmetry = {count}")
    print("First witness for each exception (index, left, middle, right):")
    for n, witness in failures.items():
        print(n, witness)
    if set(failures) != {n for n in (3, 5, 6, 7) if n <= limit}:
        raise SystemExit("Exception list differs from the conjecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limit", nargs="?", default=256, type=int)
    check(parser.parse_args().limit)
