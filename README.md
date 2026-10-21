# MahiZip

A lossless compressor built from scratch in Python + [Numba](https://numba.pydata.org/), with
two modes: one that roughly matches `xz`, and one that beats it — verified on real files, not
just toy benchmarks.

Started as a broken prototype that was 2.6x worse than `gzip`. Ended up beating `xz` by up to
23% on real full-size files, confirmed on a real multi-core machine, not just a sandbox.

## The two modes

| | **FAST** | **MAX** |
|---|---|---|
| Method | LZ77 (optimal parser) + range coder | Context mixing (11 models) + optional BWT front-end |
| Typical ratio vs `xz` | ~matches it | 2–23% smaller, file-dependent |
| Compress speed | ~0.6 MB/s | ~0.15 MB/s |
| Decompress speed | ~26 MB/s | ~0.4–4 MB/s |
| Use it for | Everyday files, fast turnaround | Archival — compress once, rarely re-touch |

Both modes share one file format, one decoder safety net (every block is CRC32'd, and
verified by compress-time round-trip by default), and one `mahizip.py c|d|t` CLI.

## Install

```
pip install numba pydivsufsort
pip install zstandard brotli   # optional, only needed by bench.py for extra comparison codecs
```

## Usage

```bash
# compress / decompress a single file
python3 mahizip.py c input.txt output.mz --profile fast          # default: fast, effort 1
python3 mahizip.py c input.bin output.mz --profile max --ram auto  # max ratio, auto RAM sizing
python3 mahizip.py d output.mz restored.txt
python3 mahizip.py t output.mz                                    # verify only, no output file

# works as a real Unix filter -- compose with tar for multi-file archives
tar cf - mydir/ | python3 mahizip.py c - archive.mzmax --profile max
python3 mahizip.py d archive.mzmax - | tar xf - -C restored/

# use multiple cores (blocks are independent; output is byte-identical to workers=1)
python3 mahizip.py c big.bin out.mz --profile fast --workers 4

# run the full correctness suite (edge cases, fuzzing, corruption detection)
python3 mahizip.py selftest
```

`--profile max` options: `--ram {tiny,small,medium,large,auto}` controls table size vs. memory
(`auto`, the default, sizes each block's tables to that block). `--effort {0,1,2,3}` controls
the FAST profile's search depth.

## Benchmarks

### Every preset, this environment, Canterbury corpus (2.79 MB)
| preset | ratio | vs xz | compress | decompress | peak RAM |
|---|---:|---:|---:|---:|---:|
| `xz -9e` | 17.27% | — | 2.12s | 0.04s | 188 MB |
| FAST e0 | 20.23% | +17.1% | 0.92s | 0.16s | 187 MB |
| FAST e1 | 19.62% | +13.6% | 1.07s | 0.13s | 187 MB |
| FAST e2 | 17.54% | +1.6% | 10.79s | 0.15s | 202 MB |
| FAST e3 | 17.17% | **-0.6%** | 26.55s | 0.14s | 203 MB |
| MAX tiny | 14.08% | **-18.5%** | 11.26s | 3.61s | 223 MB |
| MAX small | 13.90% | **-19.5%** | 11.92s | 4.18s | 289 MB |
| MAX medium | 13.85% | **-19.8%** | 12.74s | 4.07s | 553 MB |
| MAX large | 13.84% | **-19.9%** | 14.94s | 4.84s | 1,610 MB |
| MAX auto (default) | 14.08% | **-18.5%** | 11.25s | 3.70s | 224 MB |

Negative "vs xz" means smaller than xz. Past FAST e2, every preset beats xz on ratio — the
cost is compress time, and for MAX, decompress time too.

### Real scale (4-core / 32 GB Kaggle run — full files, no slicing)
| file | size | MAX | xz | smaller by |
|---|---:|---:|---:|---:|
| dickens | 10.0 MB | 22.71% | 28.16% | 19.3% |
| samba | 21.6 MB | 14.87% | 17.31% | 14.1% |
| nci | 33.6 MB | 3.72% | 4.32% | 13.9% |
| webster | 40.5 MB | 15.65% | 20.44% | **23.4%** |
| mozilla | 51.2 MB | 25.65% | 26.12% | 1.8% |

**Full 212 MB Silesia corpus** (standard benchmark), FAST profile: 24.39%, vs. `xz` 22.94% and
`bzip2` 25.85% — lands between the two, as the smaller-scale numbers predicted.

**vs. `zpaq -m5`** (a real, respected reference compressor), same 94.8 MB blob: MAX gets
17.64% against zpaq's 16.34% — roughly an 8% gap. Honest and unclosed, but it holds steady
whether tested on a few KB or at full scale, rather than being a small-file artifact.

**Multi-core:** confirmed real, not just theoretical — 2 workers ≈1.9x faster, 4 workers
≈2.1x faster on a real 94.8 MB file, byte-identical output to single-threaded either way.

## How it works

**FAST** — a hash-chain LZ77 match finder feeds a price-based optimal parser (plans the whole
parse ahead using the encoder's own learned costs, instead of greedily taking the best-looking
match at each position), then an adaptive LZMA-style binary range coder.

**MAX** — predicts each bit with 9 direct hashed contexts (orders 2/3/4/6, a word context, two
sparse/skip contexts, and an order-8 context), 2 indirect contexts (a compact bit-history pair
shared through a small adaptive table, so unrelated contexts with similar histories share a
learned curve instead of each starting cold), and a match model for long repeats. Two mixers
combine these predictions, refined by two APM/SSE stages, then coded with an integer-only
arithmetic coder — same encoder and decoder code path, so it's bit-exact on any machine. Each
block also tries a BWT front-end (via `pydivsufsort`) and keeps whichever version is smaller;
BWT helps on binary/structured data and is a no-op (never chosen) on text with long repeats,
since the match model already handles those better on its own.

## What this is *not*

- **Not an archive format on its own** — no filenames, permissions, or directory structure.
  Compose with `tar` for that (see Usage above), the same way `.tar.gz`/`.tar.xz` work.
- **No encryption.** Pipe through `gpg`/`age` if you need that, same as any tar pipeline.
- **Not battle-tested.** Safety here means ~180 fuzz and corruption test cases written for
  this project, not years of adversarial scrutiny the way `zlib`/`xz`/`zstd` have had.
- **MAX is slow.** ~0.15 MB/s to compress, well under 1 MB/s to decompress on a single core.
  Fine for "compress once, rarely touch again" — not for anything interactive.
- **Python + Numba**, not a portable C binary — first run pays a JIT-compile cost (a few
  seconds), and it needs a Python environment to run at all.

## Files

- `mahizip.py` — the library and CLI (both profiles, container format, self-test)
- `mahicm.py` — the MAX profile's context-mixing engine
- `bench.py` — benchmark harness (compares against zlib/bz2/xz/zstd/brotli)
- `kaggle_benchmark.py` — single-cell script for testing at real scale on a bigger machine

## License

[MIT](LICENSE)
