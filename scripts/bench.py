#!/usr/bin/env python3
"""Benchmark TurboVec (TurboQuant) vs FAISS OPQ on size, build time, serving speed, recall.

Grid: sizes {10k,100k,1M,10M} x dims {10,100,500}, skipping 10M x 500 (20GB raw > 24GB RAM).
Both 2-bit and 4-bit. FAISS OPQ rate-matched (m = d/2 @4bit, m = d/4 @2bit, nearest divisor).
All vectors unit-normalized -> cosine; ground truth exact via IndexFlatIP.
TurboVec needs dim multiple of 8 -> zero-pad (zeros do not change cosine ranking).
Results streamed to results/results.csv so a crash on a big cell keeps earlier rows.
"""
import os, sys, time, gc, csv, argparse, math
import numpy as np
import faiss
import turbovec

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(HERE, "results")
IDX_DIR = os.path.join(RESULTS_DIR, "_indexes")
os.makedirs(IDX_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")

# Single-thread both engines for a fair serving-speed comparison.
faiss.omp_set_num_threads(1)

SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
DIMS = [10, 100, 500]
SKIP = {(10_000_000, 500)}
BITS = [2, 4]
KMAX = 100
RECALL_KS = [1, 10, 100]
SEED = 1234
OPQ_TRAIN_SAMPLE = 12_000  # fixed OPQ/PQ training subsample (ample for nbits=8 codebooks)

FIELDS = [
    "n", "dim", "engine", "bit_width", "params",
    "build_s", "size_bytes", "bytes_per_vec",
    "qps", "lat_ms_p50", "lat_ms_p99",
    "recall@1", "recall@10", "recall@100", "n_queries",
]


def pad8(a):
    p = (-a.shape[1]) % 8
    return np.pad(a, ((0, 0), (0, p))) if p else a


def divisors(d):
    return [m for m in range(1, d + 1) if d % m == 0]


def opq_m(d, target_bits):
    """Nearest divisor of d to d*target_bits/8 (PQ nbits=8 -> m bytes/vec)."""
    want = d * target_bits / 8.0
    return min(divisors(d), key=lambda m: (abs(m - want), m))


def gen_data(n, d, n_queries, rng):
    """Clustered synthetic embeddings + held-out queries, unit-normalized."""
    n_clusters = int(min(2000, max(64, n // 500)))
    centroids = rng.standard_normal((n_clusters, d)).astype("float32")
    noise = 0.45
    assign = rng.integers(0, n_clusters, size=n)
    X = centroids[assign] + noise * rng.standard_normal((n, d)).astype("float32")
    X = X.astype("float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    qa = rng.integers(0, n_clusters, size=n_queries)
    Q = centroids[qa] + noise * rng.standard_normal((n_queries, d)).astype("float32")
    Q = Q.astype("float32")
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    return X, Q


def recall_at(approx, gt, k):
    hit = 0
    for a, g in zip(approx, gt):
        hit += len(set(a[:k].tolist()) & set(g[:k].tolist()))
    return hit / (len(approx) * k)


def time_search(search_fn, Q, k, repeats):
    # warmup
    search_fn(Q[:min(16, len(Q))], k)
    per_query = []
    idx_out = None
    t0 = time.perf_counter()
    for _ in range(repeats):
        idx_out = search_fn(Q, k)
    total = time.perf_counter() - t0
    qps = (len(Q) * repeats) / total
    # single-query latency distribution (serving pattern)
    for i in range(len(Q)):
        s = time.perf_counter()
        search_fn(Q[i:i + 1], k)
        per_query.append((time.perf_counter() - s) * 1000.0)
    p50 = float(np.percentile(per_query, 50))
    p99 = float(np.percentile(per_query, 99))
    return qps, p50, p99, idx_out


def run_cell(n, d, writer, fh):
    rng = np.random.default_rng(SEED + n + d)
    n_queries = 500 if n >= 10_000_000 else 1000
    search_repeats = 1 if n >= 1_000_000 else 3
    print(f"\n=== cell n={n} d={d} (queries={n_queries}) ===", flush=True)
    t = time.perf_counter()
    X, Q = gen_data(n, d, n_queries, rng)
    print(f"  data gen {time.perf_counter()-t:.1f}s  X={X.nbytes/1e9:.2f}GB", flush=True)

    # exact ground truth (cosine = IP on normalized)
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(d)
    flat.add(X)
    _, gt = flat.search(Q, KMAX)
    del flat
    gc.collect()
    print(f"  ground truth {time.perf_counter()-t:.1f}s", flush=True)

    Xp = pad8(X)
    Qp = pad8(Q)
    pad_dim = Xp.shape[1]

    for bw in BITS:
        # ---- TurboVec ----
        try:
            t = time.perf_counter()
            tv = turbovec.TurboQuantIndex(dim=pad_dim, bit_width=bw)
            tv.add(Xp)
            try:
                tv.prepare()
            except Exception:
                pass
            build_s = time.perf_counter() - t
            path = os.path.join(IDX_DIR, f"tv_{n}_{d}_{bw}.tv")
            tv.write(path)
            size = os.path.getsize(path)
            qps, p50, p99, out = time_search(lambda q, k: tv.search(q, k), Qp, KMAX, search_repeats)
            ti = np.asarray(out[1])
            row = dict(n=n, dim=d, engine="turbovec", bit_width=bw,
                       params=f"pad_dim={pad_dim}", build_s=round(build_s, 3),
                       size_bytes=size, bytes_per_vec=round(size / n, 2),
                       qps=round(qps, 1), lat_ms_p50=round(p50, 4), lat_ms_p99=round(p99, 4),
                       n_queries=n_queries)
            for k in RECALL_KS:
                row[f"recall@{k}"] = round(recall_at(ti, gt, k), 4)
            writer.writerow(row); fh.flush()
            print(f"  TV{bw}: build={build_s:.2f}s size={size/1e6:.1f}MB "
                  f"qps={qps:.0f} r@10={row['recall@10']}", flush=True)
            os.remove(path)
            del tv
            gc.collect()
        except Exception as e:
            print(f"  TV{bw} FAILED: {e}", flush=True)

        # ---- FAISS OPQ ----
        try:
            m = opq_m(d, bw)
            fac = f"OPQ{m},PQ{m}"
            t = time.perf_counter()
            fi = faiss.index_factory(d, fac, faiss.METRIC_INNER_PRODUCT)
            # Cap OPQ training sample: 256-centroid codebooks need ~10k points;
            # more inflates train time (esp. high m at d=500) without raising recall.
            ntrain = min(n, OPQ_TRAIN_SAMPLE)
            tr = X if n <= ntrain else X[rng.choice(n, ntrain, replace=False)]
            fi.train(tr)
            fi.add(X)
            build_s = time.perf_counter() - t
            path = os.path.join(IDX_DIR, f"opq_{n}_{d}_{bw}.faiss")
            faiss.write_index(fi, path)
            size = os.path.getsize(path)
            qps, p50, p99, out = time_search(lambda q, k: fi.search(q, k), Q, KMAX, search_repeats)
            fidx = np.asarray(out[1])
            row = dict(n=n, dim=d, engine="faiss_opq", bit_width=bw,
                       params=fac, build_s=round(build_s, 3),
                       size_bytes=size, bytes_per_vec=round(size / n, 2),
                       qps=round(qps, 1), lat_ms_p50=round(p50, 4), lat_ms_p99=round(p99, 4),
                       n_queries=n_queries)
            for k in RECALL_KS:
                row[f"recall@{k}"] = round(recall_at(fidx, gt, k), 4)
            writer.writerow(row); fh.flush()
            print(f"  OPQ{bw}({fac}): build={build_s:.2f}s size={size/1e6:.1f}MB "
                  f"qps={qps:.0f} r@10={row['recall@10']}", flush=True)
            os.remove(path)
            del fi
            gc.collect()
        except Exception as e:
            print(f"  OPQ{bw} FAILED: {e}", flush=True)

    del X, Q, Xp, Qp, gt
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10_000_000)
    ap.add_argument("--dims", type=int, nargs="*", default=DIMS)
    args = ap.parse_args()

    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader(); fh.flush()

    for n in SIZES:
        if n > args.max_n:
            continue
        for d in args.dims:
            if (n, d) in SKIP:
                print(f"\n=== SKIP n={n} d={d} (too large for RAM) ===", flush=True)
                continue
            run_cell(n, d, writer, fh)
    fh.close()
    print("\nDONE. results ->", CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
