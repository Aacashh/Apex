import math
import random
from itertools import combinations
import numpy as np

EPS = 1e-12


def histogram(hist, classes):
    v = np.zeros(classes)
    if not isinstance(hist, dict):
        return v
    for key, value in hist.items():
        try:
            k, count = (int(key), float(value))
            if 0 <= k < classes and math.isfinite(count) and (count >= 0):
                v[k] += count
        except (ValueError, TypeError):
            continue
    return v


def js_bits(p, q):
    p, q = (np.asarray(p, float), np.asarray(q, float))
    p, q = (p / p.sum(), q / q.sum())
    m = (p + q) / 2
    return float(sum((0.5 * np.sum(v[v > 0] * np.log2(v[v > 0] / m[v > 0])) for v in (p, q))))


def heterogeneity(clients, classes, seed, pair_cap=200):
    valid = [
        (c.id, histogram(c.label_histogram, classes)) for c in sorted(clients, key=lambda c: c.id)
    ]
    valid = [(cid, h / h.sum()) for cid, h in valid if h.sum() > 0]
    count = len(valid)
    total = count * (count - 1) // 2
    rng = random.Random(seed)
    if total <= pair_cap:
        pairs = list(combinations(range(count), 2))
    else:
        pairs = []
        for rank in rng.sample(range(total), pair_cap):
            lo, hi = (0, count - 1)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if mid * (2 * count - mid - 1) // 2 <= rank:
                    lo = mid
                else:
                    hi = mid
            offset = lo * (2 * count - lo - 1) // 2
            pairs.append((lo, lo + 1 + rank - offset))
    values = [js_bits(valid[i][1], valid[j][1]) for i, j in pairs]
    return dict(
        value=float(np.mean(values)) if values else 0.0,
        pairs=[(valid[i][0], valid[j][0]) for i, j in pairs],
        invalid_clients=len(clients) - count,
    )
