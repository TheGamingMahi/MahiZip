"""
MahiCM -- MAX profile engine: integer-only context mixing (lpaq-style design).

Everything is integer arithmetic, so encoder and decoder are bit-exact on any machine.
Encoder and decoder run the SAME function (cm_run) -- mode 1 encodes, mode 2 decodes -- so the
model can never drift between them.

Model:  order-0, order-1 (direct tables) + hashed orders 2,3,4,6 + word context
        (16-bit adaptive counters in nibble slots with 16-bit check, 2-way buckets)
        + 2 indirect/bit-history contexts (order-2, order-4): each context stores a compact
          (n0,n1) bit-history pair instead of its own probability; a small SHARED table then
          maps (n0,n1) -> probability and adapts globally, so unrelated contexts that happen to
          have seen "3 zeros, 1 one" share a learned curve instead of each starting from a
          neutral prior. This is the ICM/indirect technique used in lpaq/zpaq.
        + match model (longest recent match, predicted bit strength learned per length)
        -> 2 mixers (weight sets picked by partial byte / match state) averaged in the stretch domain
        -> 2 APM/SSE stages -> 32-bit carryless binary arithmetic coder (12-bit probabilities).
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- tables (integer only)
_SQ = [1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
       3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094]
SQUASH_T = np.array(_SQ, dtype=np.int64)


def _squash_py(d):
    if d > 2047:
        return 4095
    if d < -2047:
        return 0
    w = d & 127
    i = (d >> 7) + 16
    return (_SQ[i] * (128 - w) + _SQ[i + 1] * w + 64) >> 7


def _make_stretch():
    t = np.zeros(4096, dtype=np.int64)
    pi = 0
    for x in range(-2047, 2048):
        v = _squash_py(x)
        for i in range(pi, v + 1):
            t[i] = x
        pi = v + 1
    for i in range(pi, 4096):
        t[i] = 2047
    return t


STRETCH = _make_stretch()
DT = np.array([16384 // (i + i + 3) for i in range(1024)], dtype=np.int64)
# reciprocal table for 16-bit hashed counters: rate = 65536/(n+1.5)
RT = np.array([int(65536 / (n + 1.5)) for n in range(16)], dtype=np.int64)

NH = 9                  # hashed contexts: o2 o3 o4 o6 word | skip(c3c4) skip(c2c3) o8 (c1,c4th)
NICM = 2                 # indirect/bit-history contexts, on orders 2 and 4 (reuses hc[0], hc[2])
ICM_CAP = 40              # bit-history (n0, n1) saturate here before the nonstationary decay kicks in
ICM_STATES = (ICM_CAP + 1) * (ICM_CAP + 1)   # shared StateMap size per ICM context
NI = 2 + NH + NICM + 2 + 1     # inputs: o0 o1 hashed(NH) indirect(NICM) match(2) bias
MINLEN = 5


@njit(nogil=True, cache=True)
def squash(d):
    if d > 2047:
        return 4095
    if d < -2047:
        return 0
    w = d & 127
    i = (d >> 7) + 16
    return (SQUASH_T[i] * (128 - w) + SQUASH_T[i + 1] * w + 64) >> 7


@njit(nogil=True, cache=True)
def _h32(a, b):
    x = (a * 0x9E3779B1 + b * 0x85EBCA6B + 0x27D4EB2F) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x


@njit(nogil=True, cache=True)
def _dc_update(t, i, y, limit):
    c = t[i]
    n = c & 1023
    p = c >> 10
    if n < limit:
        c += 1
    else:
        c = (c & ~1023) | limit
    c += ((((y << 22) - p) >> 3) * DT[n]) & ~1023
    t[i] = c


@njit(nogil=True, cache=True)
def _lenq(l):
    if l < 16:
        return l
    q = 16 + ((l - 16) >> 3)
    return q if q < 63 else 63


@njit(nogil=True, cache=True)
def cm_run(mode, data, n, buf, tbits, cfg):
    """mode 1: encode data[:n] into buf, return bytes written (-1 on overflow)
       mode 2: decode n bytes from buf into data, return 0 (or -1 on truncated input)"""
    nslots = 1 << tbits
    mask = nslots - 1
    T = np.zeros((NH, nslots * 16), dtype=np.uint16)
    Tic = np.zeros((NICM, nslots * 16), dtype=np.uint16)     # packs (n0<<8 | n1) per nibble slot
    SM = np.full((NICM, ICM_STATES), 1 << 31, dtype=np.int64)  # shared (n0,n1) -> probability map
    t0 = np.full(256, 1 << 31, dtype=np.int64)
    t1 = np.full(65536, 1 << 31, dtype=np.int64)
    W1 = np.full((256, NI), 20000, dtype=np.int64)
    W2 = np.full((80, NI), 20000, dtype=np.int64)
    A1 = np.zeros(256 * 24, dtype=np.int64)
    A2 = np.zeros(65536 * 24, dtype=np.int64)
    for j in range(24):
        v = squash((j * 4096) // 23 - 2048) * 16
        for c in range(256):
            A1[c * 24 + j] = v
        for c in range(65536):
            A2[c * 24 + j] = v
    mmtab = np.zeros(1 << 22, dtype=np.int32)
    mmsm = np.full(64, 1 << 31, dtype=np.int64)
    x = np.zeros(NI, dtype=np.int64)
    sb = np.zeros(NH, dtype=np.int64)
    sbic = np.zeros(NICM, dtype=np.int64)
    hc = np.zeros(NH, dtype=np.int64)

    lr = cfg[0]
    lim_hi = cfg[1]
    lim0 = cfg[2]
    lim1 = cfg[3]
    apm_rate = cfg[4]

    # arithmetic coder
    x1 = np.int64(0)
    x2 = np.int64(0xFFFFFFFF)
    xv = np.int64(0)
    op = 0
    ip = 0
    if mode == 2:
        for _ in range(4):
            xv = (xv << 8) | (np.int64(buf[ip]) if ip < len(buf) else 0)
            ip += 1

    c4 = np.int64(0)      # last 4 bytes
    c8 = np.int64(0)      # the 4 before
    wh = np.int64(0)      # word hash
    mm_ptr = 0
    mm_len = 0

    for i in range(n):
        # ---- byte-level contexts
        c1 = c4 & 0xFF
        hc[0] = _h32(c4 & 0xFFFF, 2)
        hc[1] = _h32(c4 & 0xFFFFFF, 3)
        hc[2] = _h32(c4, 4)
        hc[3] = _h32(c4, (c8 & 0xFFFF) + 6 * 65536)
        hc[4] = _h32(wh, 1000 + (c1 if wh == 0 else 0))
        hc[5] = _h32((c4 >> 16) & 0xFFFF, 12)
        hc[6] = _h32((c4 >> 8) & 0xFFFF, 11)
        hc[7] = _h32(c4, c8 + 15 * 4294967296)
        hc[8] = _h32(c4 & 0xFF0000FF, 13)
        pbyte = -1
        if mm_len > 0:
            pbyte = np.int64(data[mm_ptr])
        lq = _lenq(mm_len)
        byte = np.int64(0)
        if mode == 1:
            byte = np.int64(data[i])
        c0 = 1
        nib = 1
        for bp in range(7, -1, -1):
            if bp == 7 or bp == 3:
                nib = 1
                nc = 0 if bp == 7 else c0
                for k in range(NH):
                    h1 = _h32(hc[k], nc)
                    chk = _h32(hc[k] ^ 0x5BD1E995, nc + 0x1B873593) & 0xFFFF
                    if chk == 0:
                        chk = 1
                    s0 = (h1 & mask) * 16
                    s1 = ((h1 & mask) ^ 1) * 16
                    if T[k, s0] == chk:
                        sb[k] = s0
                    elif T[k, s1] == chk:
                        sb[k] = s1
                    else:
                        v = s0
                        if (T[k, s1 + 1] & 15) < (T[k, s0 + 1] & 15):
                            v = s1
                        T[k, v] = chk
                        for j in range(1, 16):
                            T[k, v + j] = 2048 << 4
                        sb[k] = v
                icm_hc = (hc[0], hc[2])     # icm0: order-2, icm1: order-4 (same info, different modeling)
                for k in range(NICM):
                    h1 = _h32(icm_hc[k], nc + 5000)
                    chk = _h32(icm_hc[k] ^ 0x5BD1E995, nc + 0x1B873593 + 5000) & 0xFFFF
                    if chk == 0:
                        chk = 1
                    s0 = (h1 & mask) * 16
                    s1 = ((h1 & mask) ^ 1) * 16
                    if Tic[k, s0] == chk:
                        sbic[k] = s0
                    elif Tic[k, s1] == chk:
                        sbic[k] = s1
                    else:
                        v = s0
                        c0v = (Tic[k, s0 + 1] >> 8) + (Tic[k, s0 + 1] & 255)
                        c1v = (Tic[k, s1 + 1] >> 8) + (Tic[k, s1 + 1] & 255)
                        if c1v < c0v:    # fewer total observations -> evict
                            v = s1
                        Tic[k, v] = chk
                        for j in range(1, 16):
                            Tic[k, v + j] = 0     # (n0=0, n1=0): a genuinely unseen, neutral state
                        sbic[k] = v
            # ---- inputs
            x[0] = STRETCH[t0[c0] >> 20]
            x[1] = STRETCH[t1[(c1 << 8) | c0] >> 20]
            for k in range(NH):
                x[2 + k] = STRETCH[T[k, sb[k] + nib] >> 4]
            for k in range(NICM):
                v = Tic[k, sbic[k] + nib]
                st_idx = (v >> 8) * (ICM_CAP + 1) + (v & 255)
                x[2 + NH + k] = STRETCH[SM[k, st_idx] >> 20]
            exp_bit = -1
            if pbyte >= 0 and ((pbyte + 256) >> (bp + 1)) == c0:
                exp_bit = (pbyte >> bp) & 1
                st = STRETCH[mmsm[lq] >> 20]
                ll = mm_len if mm_len < 32 else 32
                if exp_bit == 1:
                    x[2 + NH + NICM] = st
                    x[3 + NH + NICM] = ll * 32
                else:
                    x[2 + NH + NICM] = -st
                    x[3 + NH + NICM] = -ll * 32
            else:
                x[2 + NH + NICM] = 0
                x[3 + NH + NICM] = 0
            x[NI - 1] = 256
            # ---- mixers
            ms2 = 0
            if exp_bit >= 0:
                ms2 = 1 + (lq >> 3)      # 1..8
                ms2 = ms2 * 8 + bp
            else:
                ms2 = bp
            d1 = np.int64(0)
            d2 = np.int64(0)
            for j in range(NI):
                d1 += W1[c0, j] * x[j]
                d2 += W2[ms2, j] * x[j]
            d1 >>= 16
            d2 >>= 16
            if d1 > 2047:
                d1 = 2047
            if d1 < -2047:
                d1 = -2047
            if d2 > 2047:
                d2 = 2047
            if d2 < -2047:
                d2 = -2047
            p1 = squash(d1)
            p2 = squash(d2)
            df = (d1 + d2) >> 1
            pf = squash(df)
            # ---- APM stages
            stf = df
            pr = (stf + 2048) * 23
            wt = pr & 0xFFF
            ba = c0 * 24 + (pr >> 12)
            pa1 = (A1[ba] * (4096 - wt) + A1[ba + 1] * wt) >> 16
            ia1 = ba + (wt >> 11)
            cx2 = c0 | (c1 << 8)
            bb = cx2 * 24 + (pr >> 12)
            pa2 = (A2[bb] * (4096 - wt) + A2[bb + 1] * wt) >> 16
            ia2 = bb + (wt >> 11)
            p = (2 * pf + pa1 + pa2 + 2) >> 2
            if p < 1:
                p = 1
            if p > 4095:
                p = 4095
            # ---- code the bit
            xmid = x1 + ((x2 - x1) >> 12) * p
            if mode == 1:
                y = (byte >> bp) & 1
            else:
                y = 1 if xv <= xmid else 0
            if y == 1:
                x2 = xmid
            else:
                x1 = xmid + 1
            while ((x1 ^ x2) & 0xFF000000) == 0:
                if mode == 1:
                    if op >= len(buf):
                        return -1
                    buf[op] = x2 >> 24
                    op += 1
                else:
                    xv = ((xv << 8) & 0xFFFFFFFF) | (np.int64(buf[ip]) if ip < len(buf) else 0)
                    ip += 1
                x1 = (x1 << 8) & 0xFFFFFFFF
                x2 = ((x2 << 8) & 0xFFFFFFFF) | 255
            # ---- update
            e1 = ((y << 12) - p1) * lr
            e2 = ((y << 12) - p2) * lr
            for j in range(NI):
                W1[c0, j] += (x[j] * e1 + 0x8000) >> 16
                W2[ms2, j] += (x[j] * e2 + 0x8000) >> 16
            _dc_update(t0, c0, y, lim0)
            _dc_update(t1, (c1 << 8) | c0, y, lim1)
            tgt = 4095 if y == 1 else 0
            for k in range(NH):
                cc = np.int64(T[k, sb[k] + nib])
                pp = cc >> 4
                nn = cc & 15
                pp += (((tgt - pp) * RT[nn]) + 32768) >> 16
                if nn < lim_hi:
                    nn += 1
                T[k, sb[k] + nib] = (pp << 4) | nn
            for k in range(NICM):
                v = Tic[k, sbic[k] + nib]
                n0v = np.int64(v >> 8)
                n1v = np.int64(v & 255)
                st_idx = n0v * (ICM_CAP + 1) + n1v
                _dc_update(SM[k], st_idx, y, lim1)
                if y == 1:
                    if n1v < ICM_CAP:
                        n1v += 1
                    if n0v > 2:
                        n0v = n0v // 2 + 1
                else:
                    if n0v < ICM_CAP:
                        n0v += 1
                    if n1v > 2:
                        n1v = n1v // 2 + 1
                Tic[k, sbic[k] + nib] = (n0v << 8) | n1v
            if exp_bit >= 0:
                _dc_update(mmsm, lq, 1 if exp_bit == y else 0, 255)
                if exp_bit != y:
                    mm_len = 0
                    pbyte = -1
            g = (y << 16) + (y << apm_rate) - y - y
            A1[ia1] += (g - A1[ia1]) >> apm_rate
            A2[ia2] += (g - A2[ia2]) >> apm_rate
            nib = nib * 2 + y
            c0 = c0 * 2 + y
        # ---- byte finished
        byte = c0 & 255
        if mode == 2:
            data[i] = byte
        c8 = ((c8 << 8) | (c4 >> 24)) & 0xFFFFFFFF
        c4 = ((c4 << 8) | byte) & 0xFFFFFFFF
        if (byte >= 65 and byte <= 90) or (byte >= 97 and byte <= 122):
            wh = (wh * 263 + (byte | 32)) & 0xFFFFFFFF
        else:
            wh = 0
        # match model
        if mm_len > 0:
            if data[mm_ptr] == byte:
                mm_len += 1
                mm_ptr += 1
            else:
                mm_len = 0
        if i + 1 >= MINLEN:
            hm = 0
            for j in range(MINLEN):
                hm = (hm * 0x2F0B4A27 + np.int64(data[i - j]) + 1) & 0xFFFFFFFF
            hm = (hm * 0x9E3779B1 & 0xFFFFFFFF) >> 10
            if mm_len == 0:
                cand = np.int64(mmtab[hm])
                if cand > 0:
                    l = 0
                    while l < 400 and cand - 1 - l >= 0 and data[cand - 1 - l] == data[i - l]:
                        l += 1
                    if l >= MINLEN:
                        mm_len = l
                        mm_ptr = cand
            mmtab[hm] = i + 1
    if mode == 1:
        for _ in range(4):
            if op >= len(buf):
                return -1
            buf[op] = x1 >> 24
            op += 1
            x1 = (x1 << 8) & 0xFFFFFFFF
        return op
    return 0


DEFAULT_CFG = np.array([20, 15, 255, 255, 5], dtype=np.int64)   # lr, hashed limit, o0 limit, o1 limit, apm rate


def tbits_for(n):
    b = 16
    while (1 << b) < 2 * n and b < 22:
        b += 1
    return b


def cm_encode(raw, cfg=None, tbits=None):
    n = len(raw)
    data = np.frombuffer(raw, dtype=np.uint8).copy()      # writable: cm_run is shared with the decoder
    buf = np.empty(n + n // 2 + 1024, dtype=np.uint8)
    r = cm_run(1, data, n, buf, tbits_for(n) if tbits is None else tbits, DEFAULT_CFG if cfg is None else cfg)
    if r < 0:
        return None
    return buf[:r].tobytes()


def cm_decode(payload, n, cfg=None, tbits=None):
    out = np.zeros(n, dtype=np.uint8)
    buf = np.frombuffer(payload, dtype=np.uint8).copy()
    r = cm_run(2, out, n, buf, tbits_for(n) if tbits is None else tbits, DEFAULT_CFG if cfg is None else cfg)
    return out if r == 0 else None
