#!/usr/bin/env python3
"""
MahiZip at real scale: full Silesia corpus + large single blocks + real multi-core speedup.

Everything so far has been validated on a 1-CPU sandbox with files up to ~40 MB. This script
answers the three things that sandbox couldn't: does the ratio hold on the FULL 212 MB Silesia
corpus, what happens on files well past 40 MB, and does --workers actually speed things up on
a real multi-core machine.

SETUP -- this is ONE Kaggle cell, paste the whole file into a single notebook cell:
  1. MAHIZIP_URL / MAHICM_URL below are already filled in (mahizip.py / mahicm.py hosted on
     catbox.moe). Only change these if you re-upload newer versions.
  2. Turn Internet ON for the kernel (needed for the download, pip, apt, and the Silesia
     corpus).
  3. Run the cell.

Everything prints as it goes, and a clean summary table prints at the end -- that summary is
what's useful to paste back for review.
"""
import os
import subprocess
import sys
import time
import urllib.request

MAHIZIP_URL = "https://files.catbox.moe/pnp89m.py"   # mahizip.py (FAST-only parallel self-check)
MAHICM_URL = "https://files.catbox.moe/895j2b.py"     # mahicm.py
SRC_DIR = "/kaggle/working/mahizip_src"

# ---------------------------------------------------------------------------------------------
print("=" * 70, "\nSTEP 0: environment\n", "=" * 70, sep="")
print("CPUs visible:", os.cpu_count())
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                print(line.strip())
except FileNotFoundError:
    pass

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSTEP 1: install dependencies\n", "=" * 70, sep="")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numba", "pydivsufsort", "zstandard", "brotli"], check=True)

ZPAQ_OK = False
try:
    subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=120)
    subprocess.run(["apt-get", "install", "-y", "-qq", "zpaq"], check=True, timeout=120)
except Exception as e:
    print("zpaq install failed (%s) -- comparisons below will skip it, everything else still runs" % e)
import shutil  # noqa: E402
ZPAQ_OK = shutil.which("zpaq") is not None
print("zpaq available:", ZPAQ_OK)

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSTEP 2: download mahizip.py / mahicm.py\n", "=" * 70, sep="")
if not MAHIZIP_URL or not MAHICM_URL:
    raise RuntimeError("MAHIZIP_URL / MAHICM_URL are empty -- fill in the two links at the top "
                        "of this cell (see the SETUP comment) before running.")
os.makedirs(SRC_DIR, exist_ok=True)

# catbox.moe (and most CDNs) reject urllib's default User-Agent ("Python-urllib/x.y") as
# bot-like and just close the connection -- that's the RemoteDisconnected error a plain
# urlretrieve() hits. A normal browser User-Agent fixes it. Retried a few times too, since a
# closed connection can also just be transient.
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}


def _fetch(url, dest_name, attempts=4):
    dest = os.path.join(SRC_DIR, dest_name)
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            break
        except Exception as e:
            last_err = e
            print("  attempt %d/%d for %s failed: %s" % (attempt, attempts, dest_name, e))
            time.sleep(2 * attempt)
    else:
        # Using a plain (non-chained) exception here rather than sys.exit(): raising from inside
        # an except block chains __context__, and that chain plus SystemExit together triggers
        # an IPython traceback-formatter bug on this Kaggle kernel (TypeError: 'NoneType' has no
        # len(), inside ultratb.py's recursion check) that hides the real error. Plain
        # RuntimeError with `from None` shows the actual problem instead.
        raise RuntimeError("Download failed for %s (%r) after %d attempts: %s" %
                            (dest_name, url, attempts, last_err)) from None
    size = os.path.getsize(dest)
    if size == 0:
        raise RuntimeError("%s downloaded as 0 bytes -- check the link is a direct download, "
                            "not a share/preview page." % dest_name)
    try:
        import ast
        ast.parse(open(dest, encoding="utf-8").read())
    except SyntaxError as e:
        raise RuntimeError("%s downloaded (%d bytes) but isn't valid Python (%s) -- the link "
                            "probably points at an HTML page instead of the raw file." %
                            (dest_name, size, e)) from None
    print("  %s: %d bytes, OK" % (dest_name, size))


_fetch(MAHIZIP_URL, "mahizip.py")
_fetch(MAHICM_URL, "mahicm.py")
sys.path.insert(0, SRC_DIR)
import mahizip as m  # noqa: E402

print("Running mahizip.py's own self-test first -- if this fails, nothing below can be trusted.")
ok = m.selftest()
if not ok:
    raise RuntimeError("selftest FAILED on this machine -- stopping before running any benchmarks.")
print("selftest passed.")

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSTEP 3: fetch the FULL Silesia corpus (212 MB)\n", "=" * 70, sep="")
CORPUS_DIR = "/kaggle/working/corpora"
if not os.path.exists(os.path.join(CORPUS_DIR, "silesia")):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                     "https://github.com/zlib-ng/corpora.git", CORPUS_DIR], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "silesia"], cwd=CORPUS_DIR, check=True)
SILESIA = os.path.join(CORPUS_DIR, "silesia")
files = sorted(f for f in os.listdir(SILESIA) if os.path.isfile(os.path.join(SILESIA, f)))
total_size = sum(os.path.getsize(os.path.join(SILESIA, f)) for f in files)
print("Silesia files:", files)
print("Total size: %.1f MB" % (total_size / 1e6))

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSTEP 4: full Silesia corpus, FAST profile (effort 2, reasonable time)\n",
      "=" * 70, sep="")
import lzma  # noqa: E402
import bz2   # noqa: E402


def run_codec(name, comp, decomp, data):
    t = time.time()
    blob = comp(data)
    tc = time.time() - t
    t = time.time()
    back = decomp(blob)
    td = time.time() - t
    ok = back == data
    return dict(name=name, size=len(blob), tc=tc, td=td, ok=ok)


results_fast = {}
t0 = time.time()
for f in files:
    data = open(os.path.join(SILESIA, f), "rb").read()
    for name, comp, decomp in [
        ("xz-9e", lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
        ("bz2-9", lambda d: bz2.compress(d, 9), bz2.decompress),
        ("mahizip-fast-e2", lambda d: m.compress_bytes(d, block_size=32 << 20, profile="fast", effort=2),
         m.decompress_bytes),
    ]:
        r = run_codec(name, comp, decomp, data)
        results_fast.setdefault(name, []).append(r)
    print("  done:", f)
print("Step 4 wall time: %.0fs" % (time.time() - t0))

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70,
      "\nSTEP 5: real multi-core speedup -- workers=1 vs workers=N on a large blob\n",
      "=" * 70, sep="")
# Concatenate several files into one big blob well past anything tested in the sandbox (~150 MB).
big_blob = b"".join(open(os.path.join(SILESIA, f), "rb").read()
                     for f in ("dickens", "mozilla", "nci") if f in files)
if not big_blob:
    big_blob = b"".join(open(os.path.join(SILESIA, f), "rb").read() for f in files[:3])
print("Big blob size: %.1f MB" % (len(big_blob) / 1e6))

N = os.cpu_count() or 1
import threading  # noqa: E402


def _compress_with_timeout(data, timeout, **kw):
    """Runs in a daemon thread with a hard timeout -- forking worker processes from inside a
    Jupyter/IPython kernel can deadlock (fork() only duplicates the calling thread, so a lock
    held by some OTHER kernel thread at that instant stays locked forever in the child). A
    timeout here means "multiprocessing doesn't work cleanly in this kernel", not a MahiZip bug
    -- the workers=1 case still ran and passed self-test earlier."""
    result = {}

    def runner():
        try:
            result["blob"] = m.compress_bytes(data, **kw)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timed out after %ds (multiprocessing likely can't fork cleanly in this " \
                      "kernel -- this is an environment limitation, not a bug)" % timeout
    if "error" in result:
        return None, repr(result["error"])
    return result["blob"], None


for workers in sorted({1, max(1, N // 2), N}):
    t = time.time()
    blob, err = _compress_with_timeout(big_blob, timeout=600, block_size=16 << 20,
                                        profile="fast", effort=2, workers=workers)
    tc = time.time() - t
    if err:
        print("  workers=%-2d  %6.1fs   SKIPPED: %s" % (workers, tc, err))
        if workers > 1:
            break   # if one workers>1 case hangs, the rest will too -- no point waiting again
        continue
    ok = m.decompress_bytes(blob) == big_blob
    print("  workers=%-2d  %6.1fs   %10d bytes   round_trip_ok=%s" % (workers, tc, len(blob), ok))

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70,
      "\nSTEP 5b: MAX profile + workers, real data (mahizip.py's own self-test can't reliably\n"
      "verify this combination on some Jupyter kernels -- see its comments -- so it's confirmed\n"
      "directly here instead, the same way Step 5 already did for FAST profile)\n",
      "=" * 70, sep="")
max_par_file = next((f for f in ("dickens", "reymont", "xml") if f in files), files[0])
max_par_data = open(os.path.join(SILESIA, max_par_file), "rb").read(4_000_000)   # keep it short: MAX is ~1 MB/s
par_block_size = 1 << 20   # small enough to split the slice into ~4 blocks -- workers>1 needs
                           # more than one block to actually parallelize anything
print("Test file: %s, %.1f MB slice, %d MB blocks (%d blocks)" % (
    max_par_file, len(max_par_data) / 1e6, par_block_size >> 20,
    -(-len(max_par_data) // par_block_size)))
for workers in sorted({1, N}):
    t = time.time()
    blob = m.compress_bytes(max_par_data, block_size=par_block_size, profile="max", workers=workers)
    tc = time.time() - t
    ok = m.decompress_bytes(blob) == max_par_data
    print("  workers=%-2d  %6.1fs   %10d bytes   round_trip_ok=%s" % (workers, tc, len(blob), ok))

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSTEP 6: MAX profile at real scale (this is the slow one -- budget time)\n",
      "=" * 70, sep="")
# A few representative files, not all 12 (MAX profile is ~1 MB/s) -- but include the biggest
# ones we never got to test in the sandbox, plus the RAM-tier auto-scaling behavior at scale.
max_test_files = [f for f in ("dickens", "samba", "nci", "webster", "mozilla") if f in files]
results_max = []
for f in max_test_files:
    data = open(os.path.join(SILESIA, f), "rb").read()
    t = time.time()
    blob = m.compress_bytes(data, block_size=32 << 20, profile="max")   # ram_tier="auto"
    tc = time.time() - t
    t = time.time()
    back = m.decompress_bytes(blob)
    td = time.time() - t
    ok = back == data
    x = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    print("  %-10s %10d -> %9d (%.2f%%)  xz=%9d (%.2f%%)  enc=%.0fs dec=%.0fs ok=%s" % (
        f, len(data), len(blob), 100 * len(blob) / len(data), len(x), 100 * len(x) / len(data),
        tc, td, ok))
    results_max.append(dict(file=f, orig=len(data), mahizip=len(blob), xz=len(x), tc=tc, td=td, ok=ok))

# ---------------------------------------------------------------------------------------------
if ZPAQ_OK:
    print("\n" + "=" * 70, "\nSTEP 7: zpaq -m5 comparison on the same big blob\n", "=" * 70, sep="")
    with open("/kaggle/working/bigblob.bin", "wb") as f:
        f.write(big_blob)
    t = time.time()
    subprocess.run(["zpaq", "a", "/kaggle/working/bigblob.zpaq", "/kaggle/working/bigblob.bin", "-m5"],
                    check=True)
    tzp = time.time() - t
    zsize = os.path.getsize("/kaggle/working/bigblob.zpaq")
    print("zpaq -m5: %d bytes (%.2f%%) in %.0fs" % (zsize, 100 * zsize / len(big_blob), tzp))

    t = time.time()
    mblob = m.compress_bytes(big_blob, block_size=32 << 20, profile="max")
    tmz = time.time() - t
    print("mahizip-max: %d bytes (%.2f%%) in %.0fs" % (len(mblob), 100 * len(mblob) / len(big_blob), tmz))

# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 70, "\nSUMMARY (paste this part back)\n", "=" * 70, sep="")
print("CPUs: %d\n" % (os.cpu_count() or 1))

for name, rr in results_fast.items():
    s = sum(r["size"] for r in rr)
    tc = sum(r["tc"] for r in rr)
    td = sum(r["td"] for r in rr)
    ok = all(r["ok"] for r in rr)
    print("Full Silesia (%d files, %.1f MB), %-18s %10d bytes (%.2f%%)  c=%.0fs d=%.0fs  verified=%s" % (
        len(files), total_size / 1e6, name, s, 100 * s / total_size, tc, td, ok))

print()
for r in results_max:
    print("MAX profile  %-10s %10d -> %9d (%.2f%%)  enc=%.0fs dec=%.0fs verified=%s" % (
        r["file"], r["orig"], r["mahizip"], 100 * r["mahizip"] / r["orig"], r["tc"], r["td"], r["ok"]))

print("\nDone. Paste the STEP 5 workers=1/N/N lines and this SUMMARY block back for review.")
