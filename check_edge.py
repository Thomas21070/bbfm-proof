"""Certify a fixed number of edge inequalities for arbitrarily large n."""

from argparse import ArgumentParser


def multiply_binomial(coeffs, weight):
    for i in range(len(coeffs) - 1, weight - 1, -1):
        coeffs[i] += coeffs[i - weight]


def first_failure(coeffs):
    for i in range(1, len(coeffs) - 1):
        if coeffs[i] ** 2 < coeffs[i - 1] * coeffs[i + 1]:
            return i
    return None


def certify(width):
    top = width + 1
    kernel = [1] + [0] * top
    for j in range(1, top + 1):
        for _ in range(4):
            multiply_binomial(kernel, j)
    if min(kernel) <= 0 or first_failure(kernel) is not None:
        raise ValueError("The four-layer kernel failed the required prefix test")
    coeffs = [1] + [0] * top
    exceptions = {}
    checked = 0
    last_failure = None
    for n in range(1, 16 * top):
        j = n
        changed = False
        while True:
            if j <= top:
                multiply_binomial(coeffs, j)
                changed = True
            if j % 2:
                break
            j //= 2
        if changed:
            checked += 1
            last_failure = first_failure(coeffs)
        if last_failure is not None:
            exceptions[n] = last_failure
    if set(exceptions) != {3, 5, 6, 7}:
        raise ValueError(exceptions)
    print(f"Edge width {width}, base parameters 1..{16 * top - 1}, "
          f"distinct prefix updates checked {checked}, exceptions {exceptions}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int, default=4096)
    args = parser.parse_args()
    if args.width < 5:
        parser.error("width must include the exception witness at index 5")
    certify(args.width)
