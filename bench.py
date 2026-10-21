#!/usr/bin/env python3
"""
MahiZip benchmark harness.

  python bench.py PATH [PATH ...] [--codecs a,b,c] [--reps 3] [--csv out.csv] [--md out.md]

PATH = file or directory (files inside are benchmarked individually, non-recursive).
Every codec is round-trip verified (byte-for-byte).  Reports per-file and corpus-total
size, ratio (% of original), compress time, decompress time (best of --reps) and
peak RSS growth during the operation.

pip install zstandard brotli   (optional; skipped if missing)
"""
import argparse
import bz2
import csv
import lzma
import os
import sys
import threading
import time
import zlib

import mahizip

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * PAGE
    except Exception:
        return 0


class PeakRSS:
    """Samples resident memory in a thread; .peak_mb = max growth over baseline."""

    def __enter__(self):
        self.base = _rss()
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, _rss())
            time.sleep(0.004)

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        self.peak = max(self.peak, _rss())

    @property
    def peak_mb(self):
        return max(0, self.peak - self.base) / 1e6


def build_codecs():
    C = {}
    C["zlib-9"] = (lambda d: zlib.compress(d, 9), zlib.decompress)
    C["bz2-9"] = (lambda d: bz2.compress(d, 9), bz2.decompress)
    C["xz-9e"] = (lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress)
    try:
        import zstandard
        C["zstd-19"] = (lambda d: zstandard.ZstdCompressor(level=19).compress(d),
                        lambda c: zstandard.ZstdDecompressor().decompress(c, max_output_size=1 << 31))
    except ImportError:
        pass
    try:
        import brotli
        C["brotli-11"] = (lambda d: brotli.compress(d, quality=11, lgwin=24), brotli.decompress)
    except ImportError:
        pass
    for e in (0, 1, 2, 3):
        C["mahizip-e%d" % e] = ((lambda d, e=e: mahizip.compress_bytes(d, block_size=32 << 20, effort=e)),
                                mahizip.decompress_bytes)
    C["mahizip-max"] = ((lambda d: mahizip.compress_bytes(d, block_size=32 << 20, profile="max")),
                        mahizip.decompress_bytes)   # ram_tier defaults to "auto"
    for tier in mahizip.CM_RAM_TIERS:
        C["mahizip-max-%s" % tier] = ((lambda d, t=tier: mahizip.compress_bytes(
            d, block_size=32 << 20, profile="max", ram_tier=t)), mahizip.decompress_bytes)
    return C


def run_one(name, comp, decomp, data, reps):
    with PeakRSS() as pc:
        t = time.perf_counter()
        blob = comp(data)
        tc = time.perf_counter() - t
    best = None
    pd_mb = 0.0
    for _ in range(reps):
        with PeakRSS() as pd:
            t = time.perf_counter()
            back = decomp(blob)
            td = time.perf_counter() - t
        pd_mb = max(pd_mb, pd.peak_mb)
        best = td if best is None else min(best, td)
    ok = (back == data)
    return dict(codec=name, size=len(blob), tc=tc, td=best, rss_c=pc.peak_mb, rss_d=pd_mb, ok=ok)


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp) and not f.startswith(".") and not f.lower().endswith(".md"):
                    files.append(fp)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--codecs", default="", help="comma list (default: all)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv")
    ap.add_argument("--md")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    allc = build_codecs()
    names = [c for c in a.codecs.split(",") if c] or list(allc)
    for n in names:
        if n not in allc:
            sys.exit("unknown codec %s (have: %s)" % (n, ", ".join(allc)))
    files = gather(a.paths)

    # warm up numba (compile/cache load) so timings are honest
    for n in names:
        if n.startswith("mahizip"):
            allc[n][1](allc[n][0](b"warm up warm up warm up" * 50))

    rows = []
    for fp in files:
        data = open(fp, "rb").read()
        if not data:
            continue
        for n in names:
            r = run_one(n, allc[n][0], allc[n][1], data, a.reps)
            r.update(file=os.path.basename(fp), orig=len(data))
            rows.append(r)
            if not a.quiet:
                print("%-16s %-12s %10d  %6.2f%%  c %6.2fs  d %6.3fs  %s" % (
                    r["file"], n, r["size"], 100.0 * r["size"] / r["orig"], r["tc"], r["td"],
                    "ok" if r["ok"] else "*** ROUND TRIP FAILED ***"), flush=True)

    # corpus totals
    tot_orig = sum({r["file"]: r["orig"] for r in rows}.values())
    lines = ["| codec | total size | % of original | compress s | decompress s | decode MB/s | peak RSS c/d (MB) | verified |",
             "|---|---:|---:|---:|---:|---:|---:|:--:|"]
    for n in names:
        rr = [r for r in rows if r["codec"] == n]
        s = sum(r["size"] for r in rr)
        tc = sum(r["tc"] for r in rr)
        td = sum(r["td"] for r in rr)
        lines.append("| %s | %s | %.2f%% | %.1f | %.2f | %.1f | %.0f / %.0f | %s |" % (
            n, format(s, ","), 100.0 * s / tot_orig, tc, td, tot_orig / 1e6 / max(td, 1e-9),
            max(r["rss_c"] for r in rr), max(r["rss_d"] for r in rr),
            "yes" if all(r["ok"] for r in rr) else "NO"))
    table = "\n".join(lines)
    print("\nFiles: %d   total original: %s bytes\n" % (len(files), format(tot_orig, ",")))
    print(table)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "orig", "codec", "size", "tc", "td", "rss_c", "rss_d", "ok"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
    if a.md:
        with open(a.md, "w") as f:
            f.write(table + "\n")
    if not all(r["ok"] for r in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
