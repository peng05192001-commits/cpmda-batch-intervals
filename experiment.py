"""Bounded correctness and computation benchmark. No training or test labels."""
from pathlib import Path
import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20260913
METHODS = ['dense', 'chunk1', 'chunk32', 'chunk256', 'implicit']
CASES = [('n200_g6', 200, 6), ('n2000_g6', 2000, 6),
         ('n20000_g6', 20000, 6), ('n2000_g200', 2000, 200)]


def write(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32)] + [
            (k, ctypes.c_size_t) for k in ['PeakWorkingSetSize', 'WorkingSetSize',
             'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
             'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_current = ctypes.windll.kernel32.GetCurrentProcess
    get_current.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    if not get_info(get_current(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {k: int(getattr(counters, k)) for k in
            ['PeakWorkingSetSize', 'WorkingSetSize', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']}


def fixture(n, g, batch=1000):
    rng = np.random.default_rng(SEED + n + g)
    counts = (np.array([5, 149, 35, 4, 5, 2], dtype=np.int64) * (n // 200)
              if g == 6 else np.full(g, n // g, dtype=np.int64))
    assert counts.sum() == n
    ptr = np.r_[0, counts.cumsum()].astype(np.int64)
    s = rng.normal(-20, 80, n).astype(np.float64)
    low = rng.normal(70, 35, (batch, g))
    high = low + np.abs(rng.normal(60, 15, (batch, g)))
    shifts = np.ascontiguousarray(np.stack([-low, high]))
    return s, ptr, shifts, (1 - 0.1) * (1 + 1 / n)


def validate_inputs(s, ptr, shifts, p):
    if len(s) == 0 or ptr[0] != 0 or ptr[-1] != len(s) or np.any(np.diff(ptr) <= 0):
        raise ValueError('Nonempty groups must partition a nonempty calibration set')
    if shifts.shape[0] != 2 or shifts.shape[2] != len(ptr) - 1:
        raise ValueError('Expected shifts with shape (2, B, G)')
    if not np.isfinite(p) or p < 0:
        raise ValueError('Invalid probability')
    if not np.isfinite(s).all() or not np.isfinite(shifts).all():
        raise ValueError('Only finite scores and shifts are supported')
    for g in range(len(ptr) - 1):
        block = s[ptr[g]:ptr[g + 1]]
        with np.errstate(over='ignore'):
            bounds = np.array([block.min() + shifts[:, :, g].min(),
                               block.max() + shifts[:, :, g].max()])
        if not np.isfinite(bounds).all():
            raise ValueError('Shifted candidates would overflow')
    return True


def prepare(method, s, ptr, p):
    if method == 'implicit':
        out = s.copy()
        for g in range(len(ptr) - 1):
            out[ptr[g]:ptr[g + 1]].sort()
        return out
    if method.startswith('chunk') and len(ptr) == 2 and p <= 1:
        h = (len(s) - 1) * p
        out = s.copy()
        out.partition([int(np.floor(h)), int(np.ceil(h))])
        return out
    return s


def lerp_array(a, b, t):
    diff = b - a
    return b - diff * (1 - t) if t >= .5 else a + diff * t


def numpy_endpoints(method, s, ptr, shifts, p):
    batch, n = shifts.shape[1], len(s)
    out = np.empty((batch, 2))
    if p > 1:
        out[:, 0], out[:, 1] = -np.inf, np.inf
        return out
    gcount = len(ptr) - 1
    if method == 'dense':
        # Same statistical operation and two dense arrays as the reference.
        # Reuse grouped inputs; omit original np.tile intermediates conservatively.
        lower, upper = np.empty((batch, n)), np.empty((batch, n))
        for side, buf in enumerate([lower, upper]):
            for g in range(gcount):
                start, end = ptr[g], ptr[g + 1]
                np.add(s[start:end], shifts[side, :, g, None], out=buf[:, start:end])
        out[:, 0] = -np.quantile(lower, p, axis=1)
        out[:, 1] = np.quantile(upper, p, axis=1)
        return out
    h = (n - 1) * p
    k0, k1 = int(np.floor(h)), int(np.ceil(h))
    t = h - k0
    if gcount == 1:
        for side in range(2):
            out[:, side] = lerp_array(s[k0] + shifts[side, :, 0],
                                      s[k1] + shifts[side, :, 0], t)
        out[:, 0] *= -1
        return out
    chunk = min(int(method[5:]), batch)
    buf = np.empty((chunk, n))
    for start in range(0, batch, chunk):
        end = min(start + chunk, batch)
        for side in range(2):
            active = buf[:end - start]
            for g in range(gcount):
                left, right = ptr[g], ptr[g + 1]
                np.add(s[left:right], shifts[side, start:end, g, None], out=active[:, left:right])
            active.partition([k0, k1], axis=1)
            out[start:end, side] = lerp_array(active[:, k0], active[:, k1], t)
    out[:, 0] *= -1
    return out


def run(method, prepared, ptr, shifts, p):
    if method == 'implicit':
        from selector import endpoints
        return endpoints(prepared, ptr, shifts, p)
    return numpy_endpoints(method, prepared, ptr, shifts, p)


def init_candidate():
    start = time.perf_counter()
    from selector import endpoints
    after_import = time.perf_counter()
    endpoints(np.array([-1., 1., 2., 3.]), np.array([0, 2, 4], dtype=np.int64),
              np.zeros((2, 1, 2)), .9)
    return {'import_seconds': after_import - start,
            'first_compile_and_tiny_call_seconds': time.perf_counter() - after_import}


def correctness():
    init = init_candidate()
    from selector import kth_shifted
    rng = np.random.default_rng(SEED)
    checked, ranks, maximum = 0, 0, 0.
    records = []
    def compare(s, ptr, shifts, p, label):
        nonlocal checked, maximum
        validate_inputs(s, ptr, shifts, p)
        expected = numpy_endpoints('dense', s, ptr, shifts, p)
        for method in METHODS[1:]:
            actual = run(method, prepare(method, s, ptr, p), ptr, shifts, p)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-10)
            finite = np.isfinite(expected)
            err = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.
            maximum = max(maximum, err)
            checked += actual.size
        records.append({'label': label, 'n': len(s), 'G': len(ptr) - 1, 'p': float(p)})
    for trial in range(90):
        g = int(rng.choice([1, 2, 6, 15]))
        counts = rng.integers(1, 17, g)
        ptr = np.r_[0, counts.cumsum()].astype(np.int64)
        s = rng.normal(-20, 50, ptr[-1])
        shifts = rng.normal(0, 60, (2, 7, g))
        if trial % 3 == 0:
            s, shifts = np.round(s / 20), np.round(shifts / 20)
        p = float(rng.choice([0., .5, .9, 1., .9 * (1 + 1 / len(s))]))
        compare(s, ptr, shifts, p, 'random_ties_or_signed')
        if trial < 15:
            ordered = prepare('implicit', s, ptr, p)
            union = np.concatenate([s[ptr[g]:ptr[g+1]] + shifts[0, 0, g] for g in range(len(ptr)-1)])
            union.sort()
            for k in range(len(union)):
                actual = kth_shifted(ordered, ptr, shifts[0, 0], k)
                assert actual == union[k], (trial, k, actual, union[k])
                ranks += 1
    for s in [np.array([0.]), np.array([-1., -1., 0., 0., 1.]),
              np.array([-1e16, -1e16 + 2, 1e16 - 2, 1e16]),
              np.array([-1e200, -1e200 + 1e185, 1e200, 1e200])]:
        ptr = np.arange(len(s) + 1, dtype=np.int64)
        shifts = np.zeros((2, 3, len(s)))
        shifts[:, 1, :] = -s
        shifts[:, 2, :] = np.spacing(s)
        for p in [0., .37, 1., .9 * (1 + 1 / len(s))]:
            compare(s, ptr, shifts, p, 'singleton_groups_cancellation_and_extreme')
    rejected = []
    for label, args in [
        ('empty', (np.array([]), np.array([0, 0]), np.zeros((2, 1, 1)), .9)),
        ('nan', (np.array([np.nan]), np.array([0, 1]), np.zeros((2, 1, 1)), .9)),
        ('overflow', (np.array([1.7e308]), np.array([0, 1]), np.full((2, 1, 1), 1.7e308), .9))]:
        try:
            validate_inputs(*args)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    write('correctness.json', {'status': 'passed', 'endpoint_values_compared': checked,
          'exact_order_statistics_checked': ranks, 'max_absolute_endpoint_difference': maximum,
          'cases': records, 'rejected_inputs': rejected, 'initialization': init})
    print(f'Correctness passed: {checked} endpoint comparisons, {ranks} exact ranks, max difference {maximum}', flush=True)


def worker(case, method):
    _, n, g = next(c for c in CASES if c[0] == case)
    start = time.perf_counter()
    s, ptr, shifts, p = fixture(n, g)
    validate_inputs(s, ptr, shifts, p)
    init = init_candidate() if method == 'implicit' else None
    ready = memory()
    calc_start = time.perf_counter()
    prepared = prepare(method, s, ptr, p)
    output = run(method, prepared, ptr, shifts, p)
    first_calc = time.perf_counter() - calc_start
    measured = memory()
    write(f'memory/{case}_{method}.json', {'case': case, 'method': method,
          'initialization': init, 'ready_memory': ready, 'after_compute_memory': measured,
          'first_compute_including_prepare_seconds': first_calc,
          'inside_worker_total_seconds': time.perf_counter() - start,
          'output_sha256': hashlib.sha256(output.tobytes()).hexdigest()})


def benchmark():
    assert json.loads((ROOT / 'correctness.json').read_text())['status'] == 'passed'
    init_candidate()
    records = []
    for name, n, g in CASES:
        print(f'Fixed computational load: {name}, B=1000. Not a prediction-quality experiment.', flush=True)
        s, ptr, shifts, p = fixture(n, g)
        validate_inputs(s, ptr, shifts, p)
        prepared, prep_times = {}, {}
        for method in METHODS:
            start = time.perf_counter()
            prepared[method] = prepare(method, s, ptr, p)
            prep_times[method] = time.perf_counter() - start
        expected = numpy_endpoints('dense', s, ptr, shifts, p)
        timing = {m: [] for m in METHODS}
        for m in METHODS:
            np.testing.assert_allclose(run(m, prepared[m], ptr, shifts, p), expected, atol=1e-10, rtol=1e-14)
        orders = []
        rng = np.random.default_rng(SEED + n + g)
        for repetition in range(5):
            order = list(rng.permutation(METHODS))
            orders.append(order)
            for method in order:
                start = time.perf_counter()
                output = run(method, prepared[method], ptr, shifts, p)
                elapsed = time.perf_counter() - start
                timing[method].append(elapsed)
                np.testing.assert_allclose(output, expected, atol=1e-10, rtol=1e-14)
        for method in METHODS:
            record = {'case': name, 'n': n, 'G': g, 'B': 1000, 'method': method,
                      'prepare_seconds': prep_times[method], 'warm_seconds': timing[method],
                      'median_warm_seconds': float(np.median(timing[method])), 'orders': orders}
            records.append(record)
        write('timings.json', records)
        print({m: round(float(np.median(timing[m])), 6) for m in METHODS}, flush=True)
        for method in METHODS:
            before = time.perf_counter()
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name, method], check=True)
            path = ROOT / f'memory/{name}_{method}.json'
            data = json.loads(path.read_text())
            data['launch_to_exit_seconds'] = time.perf_counter() - before
            write(str(path.relative_to(ROOT)), data)
        print(f'Fresh-process memory records complete: {name}', flush=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [ROOT / 'protocol.md', ROOT / 'experiment.py', ROOT / 'selector.py']}
    write('run_manifest.json', {'status': 'complete', 'platform': platform.platform(),
          'python': sys.version, 'numpy': np.__version__, 'seed': SEED,
          'model_fits': 0, 'artificial_loads': True, 'sources': sources,
          'timing_repeats': 5, 'memory_processes': 20, 'cpu': platform.processor()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--worker', nargs=2)
    args = parser.parse_args()
    if args.check:
        correctness()
    elif args.benchmark:
        benchmark()
    elif args.worker:
        worker(*args.worker)
    else:
        parser.error('Choose --check, --benchmark, or --worker CASE METHOD')
