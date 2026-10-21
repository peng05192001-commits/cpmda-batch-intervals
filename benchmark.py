"""Real-data, frozen-model end-to-end comparison; no fitting or query-label access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT
sys.path.insert(0, str(CORE))
import experiment as core
METHODS = ['chunk1', 'chunk32', 'chunk256', 'implicit']


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked(a, pattern):
    out = a.copy()
    out[:, pattern] = np.nan
    return out


def prepare_cache(method, xc, yc, xt, predict):
    start = time.perf_counter()
    cm = np.isnan(xc)
    patterns = np.unique(np.isnan(xt), axis=0)
    cache = {}
    sort_seconds = 0.
    cal_predict_seconds = 0.
    p = .9 * (1 + 1 / len(xc))
    for mask in patterns:
        begin = time.perf_counter()
        prediction = predict(masked(xc, mask))
        cal_predict_seconds += time.perf_counter() - begin
        scores = np.maximum(prediction[:, 0] - yc, yc - prediction[:, 1])
        assert np.isfinite(scores).all()
        unions, inverse, counts = np.unique(cm | mask, axis=0, return_inverse=True, return_counts=True)
        ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        grouped = np.concatenate([scores[inverse == g] for g in range(len(unions))])
        begin = time.perf_counter()
        prepared = core.prepare(method, grouped, ptr, p)
        sort_seconds += time.perf_counter() - begin
        cache[tuple(mask.tolist())] = (unions, prepared, ptr)
    total = time.perf_counter() - start
    return cache, {'total_seconds': total, 'calibration_prediction_seconds': cal_predict_seconds,
                   'method_prepare_seconds': sort_seconds, 'query_pattern_count': len(patterns),
                   'calibration_predictions_rows': len(patterns) * len(xc)}


def query(method, xt, cache, p, predict):
    start = time.perf_counter()
    patterns, inverse = np.unique(np.isnan(xt), axis=0, return_inverse=True)
    out = np.empty((len(xt), 2))
    prediction_seconds = 0.
    kernel_seconds = 0.
    prediction_rows = 0
    for index, pattern in enumerate(patterns):
        rows = np.flatnonzero(inverse == index)
        current = xt[rows]
        unions, scores, ptr = cache[tuple(pattern.tolist())]
        shifts = np.empty((2, len(rows), len(unions)))
        begin = time.perf_counter()
        for g, union in enumerate(unions):
            prediction = predict(masked(current, union))
            shifts[0, :, g] = -prediction[:, 0]
            shifts[1, :, g] = prediction[:, 1]
        prediction_seconds += time.perf_counter() - begin
        prediction_rows += len(rows) * len(unions)
        core.validate_inputs(scores, ptr, shifts, p)
        begin = time.perf_counter()
        out[rows] = core.run(method, scores, ptr, shifts, p)
        kernel_seconds += time.perf_counter() - begin
    total = time.perf_counter() - start
    assert np.isfinite(out).all()
    return out, {'total_seconds': total, 'query_masking_prediction_seconds': prediction_seconds,
                 'endpoint_kernel_seconds': kernel_seconds,
                 'other_grouping_validation_seconds': total - prediction_seconds - kernel_seconds,
                 'masked_prediction_rows': prediction_rows}


