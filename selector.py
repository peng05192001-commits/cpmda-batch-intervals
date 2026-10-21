"""Exact selection in shifted sorted groups; no model fitting.

The statistical target is the ICML 2023 CP-MDA-Nested implementation.
Sorted-list rank selection is existing algorithmic machinery, not a new theorem.
"""
import sys
from pathlib import Path
import numpy as np
from numba import njit


@njit(fastmath=False, cache=False)
def kth_shifted(s, ptr, delta, k):
    gcount = len(ptr) - 1
    lo = ptr[:-1].copy()
    hi = ptr[1:].copy()
    less_end = np.empty(gcount, dtype=np.int64)
    equal_end = np.empty(gcount, dtype=np.int64)
    values = np.empty(gcount, dtype=np.float64)
    weights = np.empty(gcount, dtype=np.int64)
    while True:
        count, total, last = 0, 0, -1
        for g in range(gcount):
            width = hi[g] - lo[g]
            if width:
                values[count] = s[lo[g] + width // 2] + delta[g]
                weights[count] = width
                total += width
                count += 1
                last = g
        if count == 1:
            return s[lo[last] + k] + delta[last]
        order = np.argsort(values[:count])
        cumulative = 0
        pivot = values[order[-1]]
        for j in order:
            cumulative += weights[j]
            if 2 * cumulative >= total:
                pivot = values[j]
                break
        less, equal_or_less = 0, 0
        for g in range(gcount):
            left, right = lo[g], hi[g]
            while left < right:
                mid = left + (right - left) // 2
                # Compare actual shifted values, not pivot - delta.
                if s[mid] + delta[g] < pivot:
                    left = mid + 1
                else:
                    right = mid
            less_end[g] = left
            less += left - lo[g]
            left, right = lo[g], hi[g]
            while left < right:
                mid = left + (right - left) // 2
                if s[mid] + delta[g] <= pivot:
                    left = mid + 1
                else:
                    right = mid
            equal_end[g] = left
            equal_or_less += left - lo[g]
        if k < less:
            hi[:] = less_end
        elif k < equal_or_less:
            return pivot
        else:
            lo[:] = equal_end
            k -= equal_or_less


@njit(fastmath=False, cache=False)
def interpolate(a, b, t):
    # Match NumPy's default linear interpolation, including its t >= .5 branch.
    difference = b - a
    if t >= 0.5:
        return b - difference * (1.0 - t)
    return a + difference * t


@njit(fastmath=False, cache=False)
def endpoints(s, ptr, shifts, p):
    batch = shifts.shape[1]
    out = np.empty((batch, 2), dtype=np.float64)
    if p > 1.0:
        out[:, 0] = -np.inf
        out[:, 1] = np.inf
        return out
    h = (len(s) - 1) * p
    k0, k1 = int(np.floor(h)), int(np.ceil(h))
    t = h - k0
    for side in range(2):
        for row in range(batch):
            a = kth_shifted(s, ptr, shifts[side, row], k0)
            b = a if k0 == k1 else kth_shifted(s, ptr, shifts[side, row], k1)
            out[row, side] = interpolate(a, b, t)
    out[:, 0] *= -1.0
    return out
