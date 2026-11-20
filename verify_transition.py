"""Rational Fourier bounds for BBFM Conjecture 4. Python standard library only."""

import argparse
import csv
from collections import Counter
from contextlib import nullcontext
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt


UNIT = 1 << 80
PI_UP = Q(22, 7)
D_GROUP = 512
H_GROUP = 13
BREAKS = tuple(map(Q, ("0.25", "0.5", "0.75", "1", "1.5", "2", "3", "4", "6", "8")))
STATS = Counter()


def ceil_div(a, b):
    return -(-a // b)


def log2_floor(a):
    v = a.numerator.bit_length() - a.denominator.bit_length()
    return v if a >= Q(2) ** v else v - 1


def sqrt_lower(a):
    return Q(isqrt(a.numerator * UNIT**2 // a.denominator), UNIT)


def sqrt_upper(a):
    return sqrt_lower(a) + Q(1, UNIT)


@lru_cache(maxsize=100000)
def exp_negative(a):
    """Integers enclosing UNIT*exp(-a). Alternating series then squaring."""
    if a < 0:
        raise ValueError(a)
    if a >= 80:
        return 0, 1
    halvings = max(0, (ceil_div(8 * a.numerator, a.denominator) - 1).bit_length())
    z = a / 2**halvings
    term_lo = term_hi = lower = upper = UNIT
    for i in range(1, 11):
        denom = z.denominator * i
        term_lo = term_lo * z.numerator // denom
        term_hi = ceil_div(term_hi * z.numerator, denom)
        if i % 2:
            lower -= term_hi
            upper -= term_lo
        else:
            # Order nine is a lower sum and order ten is an upper sum.
            if i < 10:
                lower += term_lo
            upper += term_hi
    lower, upper = max(0, lower), min(UNIT, upper)
    for _ in range(halvings):
        lower = lower**2 // UNIT
        upper = ceil_div(upper**2, UNIT)
    return lower, upper


@lru_cache(maxsize=100000)
def bernoulli(a):
    lo, hi = exp_negative(a)
    p_hi = ceil_div(UNIT * hi, UNIT + hi)
    v_lo = UNIT**2 * lo // (UNIT + lo)**2
    v_hi = ceil_div(UNIT**2 * hi, (UNIT + hi)**2)
    return p_hi, v_lo, v_hi


@lru_cache(maxsize=256)
def multiplicities(d, h, left, right):
    bins = 128 if d == D_GROUP else d
    rows = []
    for j in range(1, 32*bins + 1):
        i = j - 1 if d == D_GROUP else j
        low_offset = log2_floor(left*bins/j)
        high_offset = log2_floor(right*bins/i) if i else log2_floor(right*bins)+3
        low = max(0, h + 1 + low_offset)
        high = max(0, h + 1 + high_offset)
        if h == H_GROUP:
            low, high = min(low, h+1), max(high, h+1)
        if not high:
            break
        rows.append((i, j, low, high))
    return rows


def gaussian_tail(a, b):
    if b <= 0:
        return None
    factor = a/b + 1/(2*b*b*a)
    return factor * exp_negative(b*a*a)[1] / UNIT


def power_exponential(c, s, lower):
    """Upper bound on I^(s/2) exp(-c I) for I >= lower."""
    if c <= 0:
        return None
    where = max(lower, Q(s, 2)/c)
    return where**(s//2) * sqrt_upper(where) * exp_negative(c*where)[1] / UNIT


def assess(d, h, left, right, t0, t1):
    # d=512 represents all dyadic D>=512. h=13 represents all H>=13.
    if d != D_GROUP and h != H_GROUP and right*d*2**h <= 1025:
        return "small-n"
    if h != H_GROUP and right*2**h*t1 < 2:
        return "small-nt"
    bins = 128 if d == D_GROUP else d
    r = h + 1
    ksum = var_lo = var_hi = third = fourth = 0
    lower_weights = []
    for i,j,low,high in multiplicities(d,h,left,right):
        p_hi, _, v_hi = bernoulli(t0*i/bins)
        _, v_lo, _ = bernoulli(t1*j/bins)
        wlo, whi = v_lo*low, v_hi*high
        lower_weights.append((i,j,wlo))
        ksum += high*p_hi*j
        var_lo += wlo*i*i
        var_hi += whi*j*j
        third += whi*j**3
        fourth += whi*j**4
    norm = UNIT * r * bins
    remainder = Q(1, 10**6)
    mean_hi = remainder + Q(ksum, norm*bins)
    vlo = Q(var_lo, norm*bins**2)
    vhi = remainder + Q(var_hi, norm*bins**2)
    m3 = remainder + Q(third, norm*bins**3)
    m4 = remainder + Q(fourth, norm*bins**4)
    imin = Q(d*r)
    if d != D_GROUP:
        if h != H_GROUP and imin*d*mean_hi <= 4096:
            return "edge"
        imin = max(imin, 4096/(d*mean_hi))
    sd = sqrt_lower(imin*vlo)
    if sd*d < 3:
        return None

    def mass(w):
        cut = bins*(PI_UP/(2*w) + Q(1,d))
        index = cut.numerator // cut.denominator
        return Q(sum(wlo for i,j,wlo in lower_weights if i > index), norm)

    def damping(w):
        # Quadratic bound with the fourth-order penalty clipped per cell.
        p, q = w.numerator, w.denominator
        den = 12*(q*bins)**2
        clipped = sum(wlo*i*i*max(0, den - (p*j)**2) for i,j,wlo in lower_weights)
        b = max(Q(1,2) - w*w*m4/(24*vlo), Q(clipped, norm*bins**2*den)/(2*vhi))
        if d != D_GROUP:
            b = max(b, 2*Q(lower_weights[0][2],norm)/(PI_UP**2*d*d*vhi))
        return b

    vpower = vhi * sqrt_upper(vhi)
    cap = min(Q(8), PI_UP*d)
    cuts = [w for w in BREAKS if w < cap] + [cap]
    rates = [damping(w) for w in cuts]
    if cap < 8:
        far = Q(0)
    else:
        s = 9 if d == D_GROUP else 3
        extreme = power_exponential(mass(8),s,imin)
        if extreme is None:
            return None
        factor = Q(1,r**3) if d == D_GROUP else Q(d**3)
        far = 8*PI_UP*factor*vpower*extreme
    for a in map(Q, ("3.1","2.9","2.7","2.5")):
        for z in map(Q, ("2","2.5","3","3.5","4","4.5")):
            y = a*z/(t0*sd)
            if y >= 2:
                continue
            ehi = Q(exp_negative(z)[1], UNIT)
            multiplier = max((1-y*y/8)**-3, ((1+ehi)/(1-ehi))**3)
            if a**3*multiplier*m3 < Q("1.56")*6*sd*vlo:
                break
        else:
            continue
        subtotal = far
        for j in reversed(range(len(cuts))):
            w = cuts[j]
            initial = gaussian_tail(a,rates[j])
            if initial is not None and initial+subtotal < Q("0.71"):
                phase = a**3*multiplier*m3 / (6*sd*vlo)
                tail = initial+subtotal
                return ("integral", a, z, w,
                        ceil_div(10**6*phase.numerator, phase.denominator),
                        ceil_div(10**6*tail.numerator, tail.denominator))
            if j:
                start = cuts[j-1]
                choices = []
                g = gaussian_tail(max(a,start*sd), rates[j])
                if g is not None:
                    choices.append(g)
                b = power_exponential(mass(start),3,imin)
                if b is not None:
                    choices.append(Q(2,3)*w**3*vpower*b)
                if not choices:
                    break
                subtotal += min(choices)
    return None


def cover(d,h,left,right,t0,t1,depth,writer):
    outcome = assess(d,h,left,right,t0,t1)
    if outcome is not None:
        row = (outcome, "", "", "", "", "") if isinstance(outcome, str) else outcome
        STATS[row[0]] += 1
        if writer is not None:
            writer.writerow((d,h,left,right,t0,t1,*row))
        return
    if depth >= 5:
        raise ArithmeticError(("uncovered box", d,h,left,right,t0,t1))
    mid, tm = (left+right)/2, (t0+t1)/2
    STATS["subdivisions"] += 1
    for l,r in ((left,mid),(mid,right)):
        for a,b in ((t0,tm),(tm,t1)):
            cover(d,h,l,r,a,b,depth+1,writer)


def main(writer):
    if writer is not None:
        writer.writerow(("D_min","H_min","alpha_lo","alpha_hi","tau_lo","tau_hi",
                         "case","A","z","W","phase_upper_millionths","tail_upper_millionths"))
    for d in [1<<i for i in range(10)]:
        for h in range(14):
            for i in range(4,8):
                for j in range(4,8):
                    cover(d,h,Q(i,4),Q(i+1,4),Q(j,4),Q(j+1,4),0,writer)
        print("covered scale", d, dict(sorted(STATS.items())), flush=True)
    print("Every parameter box covered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witnesses", metavar="CSV", help="write the verified leaf boxes")
    output = parser.parse_args().witnesses
    with (open(output, "w", newline="", encoding="utf-8") if output else nullcontext()) as stream:
        main(csv.writer(stream) if stream is not None else None)
