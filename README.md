# TurboVec vs FAISS OPQ — benchmark

A reproducible benchmark comparing [**TurboVec**](https://github.com/RyanCodrai/turbovec)
(Google TurboQuant, Rust + SIMD, no-train flat index) against **FAISS OPQ** —
both the flat `OPQ,PQ` (exhaustive) and the production-realistic `OPQ,IVF,PQ`
(list-pruned) configurations — on **index size, build time, serving speed (QPS +
latency), and recall@1/@10/@100**.

**Grid:** {10k, 100k, 1M, 10M} vectors × {10, 100, 500} dims (10M×500 skipped —
exceeds 24 GB RAM), at 2-bit and 4-bit quantization rates. 66 configurations.

## Results

See [`report/REPORT.md`](report/REPORT.md) for the full write-up, per-cell
tables, head-to-head ratios, methodology, and caveats. Raw numbers:
[`results/results.csv`](results/results.csv).

**TL;DR:** TurboVec wins build time by 15×–3000× (no training step) and wins
serving QPS against flat OPQ by 2.7×–154×. Against the realistic OPQ+IVF it's
closer — TurboVec wins most cells but IVF's list-pruning overtakes it at very
large n + low dimension. Recall: TurboVec beats flat OPQ everywhere; vs OPQ+IVF
it's mixed. Recall is on **synthetic clustered data** and is directional only —
rerun on real embeddings before drawing conclusions.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install numpy faiss-cpu turbovec
.venv/bin/python scripts/bench.py       # TurboVec + flat OPQ  (writes results/results.csv)
.venv/bin/python scripts/bench_ivf.py   # OPQ+IVF              (appends)
.venv/bin/python scripts/make_report.py # regenerate tables from the CSV
```

Single-threaded; vectors unit-normalized (cosine); exact ground truth via
`IndexFlatIP`. TurboVec requires dim be a multiple of 8, so vectors are
zero-padded (does not change cosine ranking). See the report for full
methodology and the fairness controls (rate matching, OPQ training subsample,
IVF `nlist`/`nprobe`).

## License

MIT — see [`LICENSE`](LICENSE).
