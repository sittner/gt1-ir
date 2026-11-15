// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Sascha Ittner
//
// gt1conv - low-latency IR convolver using the BOSS GT-1 USB audio interface as an FX loop.
//
//   GT-1 (USB capture) --> convolver --> GT-1 (USB playback), GT-1 "Direct Mon" = OFF
//
// Audio: both directions run on the GT-1's single USB clock (no resampling). Zero algorithmic
// latency: the first DIRECT taps of the IR are a direct FIR in the audio thread, the rest is
// split into uniformly partitioned FFT segments (block 128 / 1024 / 8192), each in its own
// worker thread. A segment with block size B starts at IR offset >= 2B, which gives its worker
// B samples of time.
//
// IR selection: a config file maps GT-1 patches (U01..U99, P01..P99) and name tokens
// ("... IR3") to IR files or bypass. The GT-1 is put into editor mode via USB MIDI so it reports
// patch changes. On a change, a new convolution engine is built in the background, primed with
// the input history, warmed up and crossfaded in.
//
// build: gcc -O3 -ffast-math -mcpu=native -o gt1conv gt1conv.c -lasound -lfftw3f -lm -lpthread
#define _GNU_SOURCE
#include <alsa/asoundlib.h>
#include <ctype.h>
#include <errno.h>
#include <fftw3.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define RATE 44100
#define CH 2
#define RING (1 << 17)              // input / segment output ring (frames), ~3 s
#define RMASK (RING - 1)
#define DIRECT 256                  // taps computed as direct FIR in the audio thread
#define MAXP 128                    // max period size
#define MAXSEG 3
#define NPATCH 198                  // U01..U99 = 0..98, P01..P99 = 99..197
#define NSLOT 62                    // IR0-9, IRa-z, IRA-Z
#define NRETIRE 8

static const int seg_B[MAXSEG] = {128, 1024, 8192};
static const long seg_O[MAXSEG] = {256, 2048, 16384};
static const long seg_E[MAXSEG] = {2048, 16384, LONG_MAX};

// ---------------------------------------------------------------- options (command line)
static struct {
    const char *dev, *conf, *ir, *wisdom;
    int period, fill, mono, stats, selftest;
    double gain_db;
    int cpu_audio, cpu_seg[MAXSEG];
} opt = {
    .dev = "hw:GT1", .wisdom = "/var/tmp/gt1conv.wisdom",
    .period = 8, .fill = 16, .stats = 60,
    .cpu_audio = 3, .cpu_seg = {2, 1, 1},
};

// ---------------------------------------------------------------- configuration (file, reloadable)
typedef struct { int set, bypass; char path[1024]; double gain_db; } target_t;
typedef struct {
    char irdir[512];
    target_t def, patch[NPATCH], slot[NSLOT];
    int norm;                       // 0 off, 1 energy, 2 peak
    int trim;
    double ref_db, maxlen_ms, warmup_ms, xfade_ms, limit_db;
} config_t;
static config_t cfg;

// ---------------------------------------------------------------- shared state
static atomic_int running = 1;
static float *inring[3];            // input history: 0 = L, 1 = R, 2 = (L+R)/2
static fftwf_plan plan_fwd[MAXSEG], plan_inv[MAXSEG];
static _Atomic float rt_limit = 0.70795f;
static atomic_long rt_warmup = 2205, rt_xfade = 882;

typedef struct engine engine_t;
typedef struct {
    engine_t *eng;
    int idx, B, N, NB, NBP, K, active, prime;
    long O, k;
    float *tin, *tout;
    fftwf_complex *acc;
    fftwf_complex *H[2], *FDL[2];   // K * NBP each
    float *out[2];                  // output ring
    sem_t sem;
    pthread_t th;
    atomic_long ready;              // output valid for sample index < ready
    atomic_long work_ns_max;
    atomic_int miss;
} seg_t;

struct engine {
    int bypass, NL, lane_src[2];
    float bypass_gain;
    long irlen;
    float *irs[2];
    float hrev[2][DIRECT];          // direct-part taps, reversed
    float hist[2][DIRECT - 1 + MAXP];
    seg_t seg[MAXSEG];
    atomic_int stop;
    long n_start;
    char desc[1400];
};

static engine_t *_Atomic pending;               // control -> audio thread
static engine_t *_Atomic retired_q[NRETIRE];    // audio thread -> control
static atomic_int rt_switching;
static engine_t *rt_cur, *rt_nxt;               // owned by the audio thread

// stats
static atomic_int st_xrun, st_lim;
static atomic_long st_proc_max;
static atomic_uint st_pk_in, st_pk_out;
static atomic_long st_dmin = LONG_MAX, st_dmax;

// GT-1 state (MIDI thread -> control)
static struct {
    pthread_mutex_t mu;
    int connected, patch, name_ok, seq;
    char name[17];
    double changed_at;
} gt = { .mu = PTHREAD_MUTEX_INITIALIZER, .patch = -1 };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static long now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000L + t.tv_nsec; }
static void amax_l(atomic_long *a, long v) { if (v > atomic_load_explicit(a, memory_order_relaxed)) atomic_store_explicit(a, v, memory_order_relaxed); }
static void amin_l(atomic_long *a, long v) { if (v < atomic_load_explicit(a, memory_order_relaxed)) atomic_store_explicit(a, v, memory_order_relaxed); }
static void amax_u(atomic_uint *a, unsigned v) { if (v > atomic_load_explicit(a, memory_order_relaxed)) atomic_store_explicit(a, v, memory_order_relaxed); }

static void logmsg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr);
}

static char *trim(char *s);

static const char *patch_label(int p, char *buf) {
    if (p < 0) strcpy(buf, "?");
    else if (p < 99) snprintf(buf, 8, "U%02d", p + 1);
    else if (p < NPATCH) snprintf(buf, 8, "P%02d", p - 98);
    else snprintf(buf, 8, "#%d", p % 100000);
    return buf;
}

static void set_rt(int prio, int cpu, const char *who) {
    if (opt.selftest) prio = 0;   // selftest waits for workers anyway; never disturb a running instance
    if (cpu >= 0 && cpu < sysconf(_SC_NPROCESSORS_ONLN)) {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
        if (pthread_setaffinity_np(pthread_self(), sizeof cs, &cs)) logmsg("%s: affinity cpu%d failed", who, cpu);
    }
    if (!prio) return;
    struct sched_param sp = { .sched_priority = prio };
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp))
        logmsg("%s: SCHED_FIFO %d failed (run as root or grant rtprio)", who, prio);
}

// ---------------------------------------------------------------- WAV loading / resampling
static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t *p) { return p[0] | p[1] << 8; }

static float *load_wav(const char *path, int *nch, long *len, int *sr, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "%s: %s", path, strerror(errno)); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz > 0 ? sz : 1);
    if (sz <= 0 || fread(b, 1, sz, f) != (size_t)sz) { fclose(f); free(b); snprintf(err, errlen, "%s: read error", path); return NULL; }
    fclose(f);
    if (sz < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { snprintf(err, errlen, "%s: not a RIFF/WAVE file", path); free(b); return NULL; }
    int tag = 0, ch = 0, bits = 0; uint8_t *data = NULL; long dlen = 0;
    *sr = 0;
    for (long p = 12; p + 8 <= sz;) {
        uint32_t cs = rd32(b + p + 4); uint8_t *c = b + p + 8;
        if (!memcmp(b + p, "fmt ", 4) && cs >= 16) {
            tag = rd16(c); ch = rd16(c + 2); *sr = rd32(c + 4); bits = rd16(c + 14);
            if (tag == 0xFFFE && cs >= 26) tag = rd16(c + 24);
        } else if (!memcmp(b + p, "data", 4)) {
            data = c; dlen = cs; if (data + dlen > b + sz) dlen = b + sz - data;
        }
        p += 8 + (long)cs + (cs & 1);
    }
    if (!data || !ch || *sr <= 0) { snprintf(err, errlen, "%s: missing fmt/data chunk", path); free(b); return NULL; }
    if (!((tag == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (tag == 3 && (bits == 32 || bits == 64)))) {
        snprintf(err, errlen, "%s: unsupported format tag %d / %d bit", path, tag, bits); free(b); return NULL;
    }
    int bps = bits / 8;
    long n = dlen / (bps * ch);
    if (n <= 0) { snprintf(err, errlen, "%s: no samples", path); free(b); return NULL; }
    float *out = malloc(sizeof(float) * n * ch);
    for (long i = 0; i < n * ch; i++) {
        const uint8_t *s = data + i * bps; float v;
        if (tag == 3) { if (bits == 32) { float x; memcpy(&x, s, 4); v = x; } else { double x; memcpy(&x, s, 8); v = x; } }
        else switch (bits) {
            case 8: v = (s[0] - 128) / 128.0f; break;
            case 16: v = (int16_t)rd16(s) / 32768.0f; break;
            case 24: v = ((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0f; break;
            default: v = (int32_t)rd32(s) / 2147483648.0f;
        }
        out[i] = v;
    }
    free(b);
    *nch = ch; *len = n;
    return out;
}

static double bessel_i0(double x) { double s = 1, t = 1; for (int k = 1; k < 50; k++) { t *= (x / (2 * k)) * (x / (2 * k)); s += t; } return s; }

// windowed-sinc resampling of one channel (Kaiser-windowed sinc, kernel tabulated with
// RS_OS steps per input sample and linearly interpolated)
#define RS_OS 2048
static float *resample(const float *x, long n, int sr_in, long *nout) {
    double ratio = (double)RATE / sr_in;
    long m = (long)ceil(n * ratio);
    float *y = calloc(m, sizeof(float));
    double fc = fmin(1.0, ratio) * 0.97;   // cutoff relative to input Nyquist
    int W = (int)ceil(32 / fc);            // half width in input samples
    double beta = 9, i0b = bessel_i0(beta);
    long tl = (long)W * RS_OS + 2;
    double *tab = malloc(sizeof(double) * tl);
    for (long k = 0; k < tl; k++) {
        double d = (double)k / RS_OS, a = d / W;
        if (a >= 1) { tab[k] = 0; continue; }
        double s = d < 1e-12 ? 1 : sin(M_PI * fc * d) / (M_PI * fc * d);
        tab[k] = fc * s * bessel_i0(beta * sqrt(1 - a * a)) / i0b;
    }
    for (long j = 0; j < m; j++) {
        double t = j / ratio; long c = (long)floor(t); double acc = 0;
        for (long i = c - W + 1; i <= c + W; i++) {
            if (i < 0 || i >= n) continue;
            double pos = fabs(t - i) * RS_OS; long k = (long)pos;
            if (k + 1 >= tl) continue;
            double fr = pos - k;
            acc += x[i] * (tab[k] + fr * (tab[k + 1] - tab[k]));
        }
        y[j] = acc;
    }
    free(tab);
    *nout = m;
    return y;
}

// ---------------------------------------------------------------- IR cache (44.1 kHz, raw level)
typedef struct ircache { char path[1100]; time_t mtime; int nch, sr; long len, frames; float *ch[2]; struct ircache *next; } ircache_t;
static ircache_t *ircache;

static ircache_t *ir_get(const char *path, char *err, size_t errlen) {
    struct stat st;
    if (stat(path, &st)) { snprintf(err, errlen, "%s: %s", path, strerror(errno)); return NULL; }
    for (ircache_t **pp = &ircache; *pp; pp = &(*pp)->next) {
        ircache_t *c = *pp;
        if (strcmp(c->path, path)) continue;
        if (c->mtime == st.st_mtime) return c;
        *pp = c->next; free(c->ch[0]); free(c->ch[1]); free(c);   // file changed: reload
        break;
    }
    int ch, sr; long n;
    float *raw = load_wav(path, &ch, &n, &sr, err, errlen);
    if (!raw) return NULL;
    ircache_t *c = calloc(1, sizeof *c);
    snprintf(c->path, sizeof c->path, "%s", path);
    c->mtime = st.st_mtime; c->nch = ch >= 2 ? 2 : 1; c->sr = sr; c->frames = n;
    for (int k = 0; k < c->nch; k++) {
        float *t = malloc(sizeof(float) * n);
        for (long i = 0; i < n; i++) t[i] = raw[i * ch + k];
        if (sr != RATE) { c->ch[k] = resample(t, n, sr, &c->len); free(t); } else { c->ch[k] = t; c->len = n; }
    }
    free(raw);
    c->next = ircache; ircache = c;
    logmsg("loaded %s (%d Hz, %d ch, %ld frames%s)", path, sr, ch, n, sr != RATE ? ", resampled to 44100 Hz" : "");
    return c;
}

// normalization gain (linear) for the given channels, before user gain
static double ir_norm_gain(float **h, int nch, long len) {
    if (cfg.norm == 0) return pow(10, cfg.ref_db / 20);
    if (cfg.norm == 1) {
        double e = 0;
        for (int k = 0; k < nch; k++) { double s = 0; for (long i = 0; i < len; i++) s += (double)h[k][i] * h[k][i]; e = fmax(e, s); }
        return pow(10, cfg.ref_db / 20) / sqrt(e);
    }
    // peak: loudest 1/3-octave band (power average) between 80 Hz and 6 kHz -> ref dB
    long NFFT = 65536; while (NFFT < 2 * len) NFFT *= 2;
    float *in = fftwf_malloc(sizeof(float) * NFFT);
    fftwf_complex *out = fftwf_malloc(sizeof(fftwf_complex) * (NFFT / 2 + 1));
    double *cum = malloc(sizeof(double) * (NFFT / 2 + 2));
    fftwf_plan p = fftwf_plan_dft_r2c_1d(NFFT, in, out, FFTW_ESTIMATE);
    double best = 1e-30, df = (double)RATE / NFFT;
    for (int k = 0; k < nch; k++) {
        memset(in, 0, sizeof(float) * NFFT);
        memcpy(in, h[k], sizeof(float) * len);
        fftwf_execute(p);
        cum[0] = 0;
        for (long b = 0; b <= NFFT / 2; b++) cum[b + 1] = cum[b] + (double)out[b][0] * out[b][0] + (double)out[b][1] * out[b][1];
        for (double f = 80; f <= 6000; f *= pow(2, 1 / 24.0)) {
            long lo = (long)(f * pow(2, -1 / 6.0) / df), hi = (long)(f * pow(2, 1 / 6.0) / df);
            if (hi <= lo) hi = lo + 1;
            double avg = (cum[hi + 1] - cum[lo]) / (hi + 1 - lo);
            if (avg > best) best = avg;
        }
    }
    fftwf_destroy_plan(p); fftwf_free(in); fftwf_free(out); free(cum);
    return pow(10, cfg.ref_db / 20) / sqrt(best);
}

// ---------------------------------------------------------------- convolution engine
static void cmac(float *restrict acc, const float *restrict h, const float *restrict x, int nb) {
    for (int i = 0; i < nb; i++) {
        float hr = h[2 * i], hi = h[2 * i + 1], xr = x[2 * i], xi = x[2 * i + 1];
        acc[2 * i] += hr * xr - hi * xi;
        acc[2 * i + 1] += hr * xi + hi * xr;
    }
}

// spectrum of the overlap-save input window of block kk into its FDL slot
static void fft_block(seg_t *s, long kk) {
    engine_t *e = s->eng;
    long base = kk * s->B - s->B;
    int slot = kk % s->K;
    for (int l = 0; l < e->NL; l++) {
        const float *r = inring[e->lane_src[l]];
        for (int i = 0; i < s->N; i++) s->tin[i] = r[(base + i) & RMASK];
        fftwf_execute_dft_r2c(plan_fwd[s->idx], s->tin, s->FDL[l] + (long)slot * s->NBP);
    }
}

static void *seg_worker(void *arg) {
    seg_t *s = arg;
    engine_t *e = s->eng;
    char name[16]; snprintf(name, sizeof name, "conv-seg%d", s->idx);
    pthread_setname_np(pthread_self(), name);
    set_rt(80 - 5 * s->idx, opt.cpu_seg[s->idx], name);
    while (1) {
        sem_wait(&s->sem);
        if (atomic_load(&e->stop)) break;
        long t0 = now_ns();
        long k = s->k;
        if (s->prime) {   // fill the delay line with the input history before activation
            for (long kk = k - s->K + 1; kk < k; kk++) if (kk >= 0) fft_block(s, kk);
            s->prime = 0;
        }
        fft_block(s, k);
        int slot = k % s->K;
        long o = k * s->B + s->O;
        for (int l = 0; l < e->NL; l++) {
            memset(s->acc, 0, sizeof(fftwf_complex) * s->NB);
            for (int j = 0; j < s->K; j++) {
                int xs = (slot - j + s->K) % s->K;
                cmac((float *)s->acc, (float *)(s->H[l] + (long)j * s->NBP), (float *)(s->FDL[l] + (long)xs * s->NBP), s->NB);
            }
            fftwf_execute_dft_c2r(plan_inv[s->idx], s->acc, s->tout);
            for (int i = 0; i < s->B; i++) s->out[l][(o + i) & RMASK] = s->tout[s->B + i];
        }
        atomic_store_explicit(&s->ready, o + s->B, memory_order_release);
        s->k = k + 1;
        amax_l(&s->work_ns_max, now_ns() - t0);
    }
    return NULL;
}

static void engine_destroy(engine_t *e) {
    if (!e) return;
    atomic_store(&e->stop, 1);
    for (int si = 0; si < MAXSEG; si++) {
        seg_t *s = &e->seg[si];
        if (!s->active) continue;
        sem_post(&s->sem);
        pthread_join(s->th, NULL);
        sem_destroy(&s->sem);
        fftwf_free(s->tin); fftwf_free(s->tout); fftwf_free(s->acc);
        for (int l = 0; l < e->NL; l++) { fftwf_free(s->H[l]); fftwf_free(s->FDL[l]); free(s->out[l]); }
    }
    for (int l = 0; l < 2; l++) free(e->irs[l]);
    free(e);
}

static void resolve_path(const char *p, char *out, size_t n) {
    if (p[0] == '/' || !cfg.irdir[0]) snprintf(out, n, "%s", p);
    else if ((size_t)snprintf(out, n, "%s/%s", cfg.irdir, p) >= n) out[n - 1] = 0;
}

static engine_t *engine_build(const target_t *t, char *err, size_t errlen) {
    engine_t *e = calloc(1, sizeof *e);
    if (t->bypass) {
        e->bypass = 1; e->bypass_gain = pow(10, t->gain_db / 20);
        e->NL = opt.mono ? 1 : 2;
        e->lane_src[0] = opt.mono ? 2 : 0; e->lane_src[1] = opt.mono ? 2 : 1;
        snprintf(e->desc, sizeof e->desc, "bypass (gain %+.1f dB)", t->gain_db);
        return e;
    }
    char full[1100]; resolve_path(t->path, full, sizeof full);
    ircache_t *c = ir_get(full, err, errlen);
    if (!c) { free(e); return NULL; }

    float pk = 0;
    for (int k = 0; k < c->nch; k++) for (long i = 0; i < c->len; i++) pk = fmaxf(pk, fabsf(c->ch[k][i]));
    if (pk <= 0) { snprintf(err, errlen, "%s: IR is silent", full); free(e); return NULL; }
    long onset = c->len;
    for (int k = 0; k < c->nch; k++) for (long i = 0; i < c->len; i++) if (fabsf(c->ch[k][i]) > pk * 0.01f) { if (i < onset) onset = i; break; }
    long skip = (cfg.trim && onset > 2) ? onset - 2 : 0;
    long len = c->len - skip, maxlen = (long)(cfg.maxlen_ms * RATE / 1000);
    if (len > maxlen) len = maxlen;
    float *src[2] = { c->ch[0] + skip, c->ch[c->nch - 1] + skip };
    double ng = ir_norm_gain(src, c->nch, len);
    double g = ng * pow(10, t->gain_db / 20);

    e->NL = opt.mono ? c->nch : 2;
    e->lane_src[0] = opt.mono ? 2 : 0; e->lane_src[1] = opt.mono ? 2 : 1;
    e->irlen = len;
    for (int l = 0; l < e->NL; l++) {
        const float *h = src[c->nch == 2 ? l : 0];
        e->irs[l] = calloc(len, sizeof(float));
        for (long i = 0; i < len; i++) e->irs[l][i] = h[i] * g;
        if (c->len - skip > len) {   // truncated: short fade-out
            int fl = len < 256 ? len : 256;
            for (int i = 0; i < fl; i++) e->irs[l][len - 1 - i] *= (float)i / fl;
        }
        for (int u = 0; u < DIRECT; u++) { long tap = DIRECT - 1 - u; e->hrev[l][u] = tap < len ? e->irs[l][tap] : 0; }
    }
    for (int si = 0; si < MAXSEG; si++) {
        seg_t *s = &e->seg[si];
        s->eng = e; s->idx = si; s->B = seg_B[si]; s->O = seg_O[si];
        if (len <= s->O) continue;
        long end = len < seg_E[si] ? len : seg_E[si];
        s->K = (end - s->O + s->B - 1) / s->B;
        s->N = 2 * s->B; s->NB = s->B + 1; s->NBP = (s->NB + 1) & ~1;
        s->active = 1;
        s->tin = fftwf_malloc(sizeof(float) * s->N);
        s->tout = fftwf_malloc(sizeof(float) * s->N);
        s->acc = fftwf_malloc(sizeof(fftwf_complex) * s->NBP);
        for (int l = 0; l < e->NL; l++) {
            s->H[l] = fftwf_malloc(sizeof(fftwf_complex) * s->NBP * s->K);
            s->FDL[l] = fftwf_malloc(sizeof(fftwf_complex) * s->NBP * s->K);
            memset(s->FDL[l], 0, sizeof(fftwf_complex) * s->NBP * s->K);
            s->out[l] = calloc(RING, sizeof(float));
            for (int j = 0; j < s->K; j++) {
                memset(s->tin, 0, sizeof(float) * s->N);
                for (int i = 0; i < s->B; i++) { long tap = s->O + (long)j * s->B + i; if (tap < len) s->tin[i] = e->irs[l][tap]; }
                fftwf_complex *Hj = s->H[l] + (long)j * s->NBP;
                fftwf_execute_dft_r2c(plan_fwd[si], s->tin, Hj);
                for (int i = 0; i < s->NB; i++) { Hj[i][0] /= s->N; Hj[i][1] /= s->N; }
            }
        }
        sem_init(&s->sem, 0, 0);
    }
    const char *base = strrchr(full, '/'); base = base ? base + 1 : full;
    snprintf(e->desc, sizeof e->desc, "%s (%ld taps, norm %+.1f dB, gain %+.1f dB%s)",
             base, len, 20 * log10(ng), t->gain_db, skip ? ", trimmed" : "");
    for (int si = 0; si < MAXSEG; si++) if (e->seg[si].active) pthread_create(&e->seg[si].th, NULL, seg_worker, &e->seg[si]);
    return e;
}

// called by the audio thread right before the engine's first period at sample index n
static void engine_activate(engine_t *e, long n) {
    e->n_start = n;
    if (e->bypass) return;
    for (int l = 0; l < e->NL; l++)
        for (int j = 0; j < DIRECT - 1; j++) e->hist[l][j] = inring[e->lane_src[l]][(n - (DIRECT - 1) + j) & RMASK];
    for (int si = 0; si < MAXSEG; si++) {
        seg_t *s = &e->seg[si];
        if (!s->active) continue;
        s->k = n / s->B;
        s->prime = 1;
        atomic_store(&s->ready, s->k * s->B + s->O);
    }
}

static void engine_post(engine_t *e, long n_end) {
    if (e->bypass) return;
    for (int si = 0; si < MAXSEG; si++)
        if (e->seg[si].active && n_end % e->seg[si].B == 0) sem_post(&e->seg[si].sem);
}

// x: per source (L, R, M), y: per lane
static void engine_process(engine_t *e, float x[3][MAXP], int P, long n, int wait, float y[2][MAXP]) {
    if (e->bypass) {
        for (int l = 0; l < e->NL; l++) for (int i = 0; i < P; i++) y[l][i] = x[e->lane_src[l]][i] * e->bypass_gain;
        return;
    }
    for (int l = 0; l < e->NL; l++) {
        float *hb = e->hist[l];
        memcpy(hb + DIRECT - 1, x[e->lane_src[l]], sizeof(float) * P);
        for (int i = 0; i < P; i++) {
            const float *w = hb + i; float acc = 0;
            for (int u = 0; u < DIRECT; u++) acc += e->hrev[l][u] * w[u];
            y[l][i] = acc;
        }
        memmove(hb, hb + P, sizeof(float) * (DIRECT - 1));
    }
    for (int si = 0; si < MAXSEG; si++) {
        seg_t *s = &e->seg[si];
        if (!s->active) continue;
        if (wait) while (atomic_load_explicit(&s->ready, memory_order_acquire) < n + P) sched_yield();
        if (atomic_load_explicit(&s->ready, memory_order_acquire) >= n + P) {
            for (int l = 0; l < e->NL; l++)
                for (int i = 0; i < P; i++) y[l][i] += s->out[l][(n + i) & RMASK];
        } else atomic_fetch_add_explicit(&s->miss, 1, memory_order_relaxed);
    }
}

static void input_store(float x[3][MAXP], int P, long n) {
    for (int c = 0; c < 3; c++) for (int i = 0; i < P; i++) inring[c][(n + i) & RMASK] = x[c][i];
}

// ---------------------------------------------------------------- ALSA / audio thread
static snd_pcm_t *open_pcm(snd_pcm_stream_t st, int period, int nper, int start_thr, int quiet) {
    snd_pcm_t *h; int e;
    if ((e = snd_pcm_open(&h, opt.dev, st, 0)) < 0) { if (!quiet) logmsg("open %s: %s", opt.dev, snd_strerror(e)); return NULL; }
    snd_pcm_hw_params_t *hw; snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(h, hw);
    snd_pcm_hw_params_set_access(h, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    if ((e = snd_pcm_hw_params_set_format(h, hw, SND_PCM_FORMAT_S32_LE)) < 0 ||
        (e = snd_pcm_hw_params_set_channels(h, hw, CH)) < 0 ||
        (e = snd_pcm_hw_params_set_rate(h, hw, RATE, 0)) < 0 ||
        (e = snd_pcm_hw_params_set_period_size(h, hw, period, 0)) < 0) {
        logmsg("%s: unsupported params: %s", opt.dev, snd_strerror(e)); snd_pcm_close(h); return NULL;
    }
    unsigned np = nper; snd_pcm_hw_params_set_periods_near(h, hw, &np, 0);
    if ((e = snd_pcm_hw_params(h, hw)) < 0) { logmsg("hw_params: %s", snd_strerror(e)); snd_pcm_close(h); return NULL; }
    snd_pcm_uframes_t bs; snd_pcm_hw_params_get_buffer_size(hw, &bs);
    snd_pcm_sw_params_t *sw; snd_pcm_sw_params_alloca(&sw);
    snd_pcm_sw_params_current(h, sw);
    snd_pcm_sw_params_set_avail_min(h, sw, period);
    snd_pcm_sw_params_set_start_threshold(h, sw, start_thr);
    snd_pcm_sw_params(h, sw);
    logmsg("%s %s: period %d, buffer %lu", opt.dev, st == SND_PCM_STREAM_PLAYBACK ? "playback" : "capture", period, bs);
    return h;
}

static void retire(engine_t *e) {
    for (int i = 0; i < NRETIRE; i++) {
        engine_t *expect = NULL;
        if (atomic_compare_exchange_strong(&retired_q[i], &expect, e)) return;
    }
    // queue full (should not happen): leak rather than free in the audio thread
}

static inline float softclip(float v, float t, int *lim) {
    float a = fabsf(v);
    if (a <= t) return v;
    (*lim)++;
    float r = 1.0f - t, c = t + r * tanhf((a - t) / r);
    return v < 0 ? -c : c;
}

static void *audio_thread(void *arg) {
    (void)arg;
    pthread_setname_np(pthread_self(), "conv-audio");
    const int P = opt.period, fill = opt.fill;
    snd_pcm_t *cap = NULL, *pb = NULL;
    for (int tries = 0; atomic_load(&running); tries++) {   // wait for the device
        if ((cap = open_pcm(SND_PCM_STREAM_CAPTURE, P, 1024 / P, 1, tries > 0)) &&
            (pb = open_pcm(SND_PCM_STREAM_PLAYBACK, P, (fill * 4 > 256 ? fill * 4 : 256) / P, fill, tries > 0))) break;
        if (cap) { snd_pcm_close(cap); cap = NULL; }
        if (tries == 0) logmsg("waiting for %s ...", opt.dev);
        sleep(1);
    }
    if (!pb) return NULL;
    set_rt(85, opt.cpu_audio, "audio");

    int32_t ibuf[MAXP * CH], obuf[MAXP * CH], zero[4096 * CH];
    memset(zero, 0, sizeof zero);
    float x[3][MAXP], yc[2][MAXP], yn[2][MAXP], out[2][MAXP];
    long n = 0;
    int fails = 0, stable = 0;

restart:
    if (fails > 10) { logmsg("audio: too many failures, exiting"); atomic_store(&running, 0); goto done; }
    snd_pcm_drop(pb); snd_pcm_drop(cap);
    if (snd_pcm_prepare(pb) < 0 || snd_pcm_prepare(cap) < 0) { logmsg("audio: device lost"); atomic_store(&running, 0); goto done; }
    snd_pcm_start(cap);
    if (snd_pcm_readi(cap, ibuf, P) < 0) { fails++; goto restart; }
    { snd_pcm_sframes_t a = snd_pcm_avail(cap); while (a >= P) { snd_pcm_readi(cap, ibuf, P); a -= P; } }
    snd_pcm_writei(pb, zero, fill);
    stable = 0;

    while (atomic_load_explicit(&running, memory_order_relaxed)) {
        snd_pcm_sframes_t r = snd_pcm_readi(cap, ibuf, P);
        if (r != P) {
            atomic_fetch_add(&st_xrun, 1); fails++;
            if (r == -ENODEV) { logmsg("audio: device lost"); atomic_store(&running, 0); break; }
            logmsg("capture: %s, restarting", r < 0 ? snd_strerror(r) : "short read");
            goto restart;
        }
        long t0 = now_ns();
        unsigned pin = 0, pout = 0; int lim = 0;
        for (int i = 0; i < P; i++) {
            int32_t a = ibuf[2 * i], b = ibuf[2 * i + 1];
            unsigned m = (unsigned)abs(a >> 1) > (unsigned)abs(b >> 1) ? (unsigned)abs(a >> 1) : (unsigned)abs(b >> 1);
            if (m > pin) pin = m;
            x[0][i] = a * (1.0f / 2147483648.0f);
            x[1][i] = b * (1.0f / 2147483648.0f);
            x[2][i] = 0.5f * (x[0][i] + x[1][i]);
        }
        input_store(x, P, n);

        if (!rt_nxt) {
            engine_t *p = atomic_exchange(&pending, NULL);
            if (p) {
                engine_activate(p, n);
                if (!rt_cur) rt_cur = p;
                else { rt_nxt = p; atomic_store(&rt_switching, 1); }
            }
        }
        if (rt_cur) engine_post(rt_cur, n + P);
        if (rt_nxt) engine_post(rt_nxt, n + P);

        if (rt_cur) {
            engine_process(rt_cur, x, P, n, 0, yc);
            for (int c = 0; c < CH; c++) memcpy(out[c], yc[rt_cur->NL == 2 ? c : 0], sizeof(float) * P);
        } else memset(out, 0, sizeof out);
        if (rt_nxt) {
            engine_process(rt_nxt, x, P, n, 0, yn);
            long W = atomic_load_explicit(&rt_warmup, memory_order_relaxed), F = atomic_load_explicit(&rt_xfade, memory_order_relaxed);
            long t = n - rt_nxt->n_start;
            for (int i = 0; i < P; i++) {
                long tt = t + i;
                float a = tt < W ? 0.0f : tt >= W + F ? 1.0f : 0.5f - 0.5f * cosf((float)M_PI * (tt - W) / F);
                for (int c = 0; c < CH; c++) out[c][i] = (1 - a) * out[c][i] + a * yn[rt_nxt->NL == 2 ? c : 0][i];
            }
            if (t + P >= W + F) {
                retire(rt_cur);
                rt_cur = rt_nxt; rt_nxt = NULL;
                atomic_store(&rt_switching, 0);
            }
        }
        float thr = atomic_load_explicit(&rt_limit, memory_order_relaxed);
        for (int i = 0; i < P; i++) {
            for (int c = 0; c < CH; c++) {
                float v = softclip(out[c][i], thr, &lim);
                if (v > 0.99999988f) v = 0.99999988f; else if (v < -1.0f) v = -1.0f;
                int32_t s = (int32_t)(v * 2147483648.0f);
                obuf[2 * i + c] = s;
                unsigned m = (unsigned)abs(s >> 1); if (m > pout) pout = m;
            }
        }
        n += P;
        if (lim) atomic_fetch_add_explicit(&st_lim, lim, memory_order_relaxed);
        amax_u(&st_pk_in, pin); amax_u(&st_pk_out, pout);
        amax_l(&st_proc_max, now_ns() - t0);

        snd_pcm_sframes_t w = snd_pcm_writei(pb, obuf, P);
        if (w != P) {
            atomic_fetch_add(&st_xrun, 1); fails++;
            if (w == -ENODEV) { logmsg("audio: device lost"); atomic_store(&running, 0); break; }
            logmsg("playback: %s, restarting", w < 0 ? snd_strerror(w) : "short write");
            goto restart;
        }
        snd_pcm_sframes_t d;
        if (snd_pcm_delay(pb, &d) == 0) { amin_l(&st_dmin, d); amax_l(&st_dmax, d); }
        if (!stable && ++stable * P > RATE) fails = 0;   // 1 s without trouble resets the failure counter
    }
done:
    snd_pcm_drop(pb); snd_pcm_drop(cap);
    snd_pcm_close(pb); snd_pcm_close(cap);
    return NULL;
}

// ---------------------------------------------------------------- GT-1 MIDI (editor mode)
static snd_rawmidi_t *midi_in, *midi_out;
static char stored_names[NPATCH][17];
static int stored_ok[NPATCH];

static void midi_sx(int cmd, const uint8_t a[4], const uint8_t *d, int dl) {
    uint8_t m[64]; int k, sum = 0;
    const uint8_t hdr[] = {0xF0, 0x41, 0x00, 0x00, 0x00, 0x00, 0x30};
    memcpy(m, hdr, sizeof hdr); k = sizeof hdr; m[k++] = cmd;
    for (int i = 0; i < 4; i++) { m[k++] = a[i]; sum += a[i]; }
    for (int i = 0; i < dl; i++) { m[k++] = d[i]; sum += d[i]; }
    m[k++] = (128 - sum % 128) % 128; m[k++] = 0xF7;
    snd_rawmidi_write(midi_out, m, k);
}
static void midi_rq1(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t size) {
    const uint8_t a[4] = {a0, a1, a2, a3}, d[4] = {0, 0, 0, size};
    midi_sx(0x11, a, d, 4);
}
static void midi_editor_mode(int on) {
    const uint8_t a[4] = {0x7F, 0, 0, 1}, d[1] = {(uint8_t)(on ? 1 : 0)};
    midi_sx(0x12, a, d, 1);
}

static void clean_name(char *dst, const uint8_t *src) {
    for (int i = 0; i < 16; i++) dst[i] = (src[i] < 32 || src[i] > 126) ? ' ' : src[i];
    dst[16] = 0;
}

static void midi_handle(const uint8_t *m, int len) {
    static const uint8_t hdr[8] = {0xF0, 0x41, 0x00, 0x00, 0x00, 0x00, 0x30, 0x12};
    if (len < 14 || memcmp(m, hdr, 8)) return;
    const uint8_t *a = m + 8, *d = m + 12; int dl = len - 14;
    char lb[8];
    if (a[0] == 0 && a[1] == 1 && a[2] == 0 && a[3] == 0 && dl >= 2) {
        int p = d[0] * 128 + d[1];
        pthread_mutex_lock(&gt.mu);
        gt.patch = p; gt.name_ok = 0; gt.seq++; gt.changed_at = now();
        pthread_mutex_unlock(&gt.mu);
        logmsg("midi: patch change -> %s (raw %02X %02X)", patch_label(p, lb), d[0], d[1]);
        midi_rq1(0x60, 0, 0, 0, 16);
    } else if (a[0] == 0x60 && a[1] == 0 && a[2] == 0 && a[3] == 0 && dl >= 16) {
        char nm[17]; clean_name(nm, d);
        pthread_mutex_lock(&gt.mu);
        if (!gt.name_ok || strcmp(nm, gt.name)) { strcpy(gt.name, nm); gt.seq++; }
        gt.name_ok = 1;
        pthread_mutex_unlock(&gt.mu);
    } else if ((a[0] == 0x10 || a[0] == 0x20) && a[1] < 99 && a[2] == 0 && a[3] == 0 && dl >= 16) {
        int p = (a[0] == 0x10 ? 0 : 99) + a[1];
        clean_name(stored_names[p], d);
        stored_ok[p] = 1;
    }
}

// read and dispatch incoming MIDI for up to timeout_ms; -1 on device error
static int midi_pump(int timeout_ms) {
    static uint8_t sx[512]; static int sxlen = -1;
    struct pollfd pfd[4];
    int np = snd_rawmidi_poll_descriptors(midi_in, pfd, 4);
    int r = poll(pfd, np, timeout_ms);
    if (r < 0) return errno == EINTR ? 0 : -1;
    for (int i = 0; i < np; i++) if (pfd[i].revents & (POLLERR | POLLHUP | POLLNVAL)) return -1;
    uint8_t buf[1024];
    for (;;) {
        ssize_t n = snd_rawmidi_read(midi_in, buf, sizeof buf);
        if (n == -EAGAIN || n == 0) break;
        if (n < 0) return -1;
        for (ssize_t i = 0; i < n; i++) {
            uint8_t b = buf[i];
            if (b >= 0xF8) continue;
            if (b == 0xF0) { sxlen = 0; sx[sxlen++] = b; }
            else if (sxlen >= 0) {
                if (sxlen < (int)sizeof sx) sx[sxlen++] = b;
                if (b == 0xF7) { midi_handle(sx, sxlen); sxlen = -1; }
                else if (b & 0x80) sxlen = -1;
            }
        }
    }
    return 0;
}

// pump until *flag is set (flag written by midi_handle in this thread); -1 error, 1 timeout
static int midi_wait(const int *flag, double timeout) {
    double t0 = now();
    while (!*flag && now() - t0 < timeout) if (midi_pump(20) < 0) return -1;
    return *flag ? 0 : 1;
}

static void *midi_thread(void *arg) {
    (void)arg;
    pthread_setname_np(pthread_self(), "conv-midi");
    char dev[96];
    snprintf(dev, sizeof dev, strchr(opt.dev, ',') ? "%s" : "%s,0,0", opt.dev);
    int warned = 0;
    while (atomic_load(&running)) {
        if (snd_rawmidi_open(&midi_in, &midi_out, dev, SND_RAWMIDI_NONBLOCK) < 0) {
            if (!warned++) logmsg("midi: cannot open %s, retrying", dev);
            for (int i = 0; i < 20 && atomic_load(&running); i++) usleep(100000);
            continue;
        }
        warned = 0;
        midi_editor_mode(1);
        pthread_mutex_lock(&gt.mu); gt.connected = 1; gt.name_ok = 0; pthread_mutex_unlock(&gt.mu);
        logmsg("midi: connected to %s, editor mode on", dev);

        // identify the current patch by comparing its name with the stored patch names
        int dead = 0, name_flag;
        midi_rq1(0x60, 0, 0, 0, 16);
        {
            double t0 = now();
            do {
                if (midi_pump(20) < 0) { dead = 1; break; }
                pthread_mutex_lock(&gt.mu); name_flag = gt.name_ok; pthread_mutex_unlock(&gt.mu);
            } while (!name_flag && now() - t0 < 1.0);
        }
        memset(stored_ok, 0, sizeof stored_ok);
        for (int p = 0; p < NPATCH && !dead && atomic_load(&running); p++) {
            midi_rq1(p < 99 ? 0x10 : 0x20, p % 99, 0, 0, 16);
            if (midi_wait(&stored_ok[p], 0.3) < 0) dead = 1;
        }
        if (!dead) {
            pthread_mutex_lock(&gt.mu);
            char name[17]; strcpy(name, gt.name); int known = gt.patch >= 0, ok = gt.name_ok;
            pthread_mutex_unlock(&gt.mu);
            int nu = 0, npr = 0, fu = -1, fp = -1, nread = 0;
            for (int p = 0; p < NPATCH; p++) nread += stored_ok[p];
            for (int p = 0; p < NPATCH && ok; p++) if (stored_ok[p] && !strcmp(stored_names[p], name)) { if (p < 99) { nu++; fu = p; } else { npr++; fp = p; } }
            int found = nu == 1 ? fu : (nu == 0 && npr == 1) ? fp : -1;
            char lb[8], nm[17]; strcpy(nm, name); trim(nm);
            if (!known) {
                if (found >= 0) {
                    pthread_mutex_lock(&gt.mu); if (gt.patch < 0) { gt.patch = found; gt.seq++; } pthread_mutex_unlock(&gt.mu);
                    logmsg("midi: current patch '%s' identified as %s (%d names read)", nm, patch_label(found, lb), nread);
                } else logmsg("midi: current patch '%s' not uniquely identified (%d user / %d preset matches, %d names read); waiting for a patch change",
                              nm, nu, npr, nread);
            }
        }
        double last_poll = now();
        while (!dead && atomic_load(&running)) {
            if (midi_pump(100) < 0) { dead = 1; break; }
            if (now() - last_poll > 1.0) { midi_rq1(0x60, 0, 0, 0, 16); last_poll = now(); }
        }
        if (!dead) { midi_editor_mode(0); snd_rawmidi_drain(midi_out); logmsg("midi: editor mode off"); }
        else logmsg("midi: device lost");
        pthread_mutex_lock(&gt.mu); gt.connected = 0; pthread_mutex_unlock(&gt.mu);
        snd_rawmidi_close(midi_in); snd_rawmidi_close(midi_out);
        midi_in = midi_out = NULL;
    }
    return NULL;
}

// ---------------------------------------------------------------- config file
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}

static int all_space(const char *s) { while (*s) if (!isspace((unsigned char)*s++)) return 0; return 1; }

static int slot_index(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return 10 + c - 'a';
    if (c >= 'A' && c <= 'Z') return 36 + c - 'A';
    return -1;
}

// value: bypass | <file> | "<file>", optionally followed by: gain <dB>
static int parse_target(char *v, target_t *t, char *err, size_t errlen) {
    memset(t, 0, sizeof *t); t->set = 1;
    v = trim(v);
    char path[1024] = ""; char *rest = "";
    if (*v == '"') {
        char *q = strchr(v + 1, '"');
        if (!q) { snprintf(err, errlen, "missing closing quote"); return -1; }
        *q = 0; snprintf(path, sizeof path, "%s", v + 1); rest = q + 1;
    } else {
        char *g = NULL;   // last " gain <number>" at the end of the value
        for (char *p = v; (p = strstr(p, "gain")); p += 4) {
            if (!((p == v || isspace((unsigned char)p[-1])) && isspace((unsigned char)p[4]))) continue;
            char *e; strtod(p + 4, &e);
            if (e != p + 4 && all_space(e)) g = p;
        }
        if (g) { size_t pl = g - v; if (pl >= sizeof path) pl = sizeof path - 1; memcpy(path, v, pl); path[pl] = 0; rest = g; }
        else snprintf(path, sizeof path, "%s", v);
    }
    rest = trim(rest);
    if (*rest) {
        if (strncmp(rest, "gain", 4) || !isspace((unsigned char)rest[4])) { snprintf(err, errlen, "expected 'gain <dB>', got '%s'", rest); return -1; }
        char *e; t->gain_db = strtod(rest + 4, &e);
        if (e == rest + 4 || !all_space(e)) { snprintf(err, errlen, "bad gain value '%s'", rest + 4); return -1; }
    }
    char *pp = trim(path);
    if (!strcmp(pp, "bypass")) t->bypass = 1;
    else if (!*pp) { snprintf(err, errlen, "missing IR file (or 'bypass')"); return -1; }
    else snprintf(t->path, sizeof t->path, "%s", pp);
    return 0;
}

static int parse_patch_key(const char *k) {
    char c = toupper((unsigned char)k[0]);
    if ((c == 'U' || c == 'P') && isdigit((unsigned char)k[1])) {
        char *e; long v = strtol(k + 1, &e, 10);
        if (*e || v < 1 || v > 99) return -1;
        return (c == 'U' ? 0 : 99) + (int)v - 1;
    }
    return -1;
}

static void cfg_defaults(config_t *c) {
    memset(c, 0, sizeof *c);
    c->norm = 2; c->ref_db = 0; c->maxlen_ms = 1000; c->warmup_ms = 50; c->xfade_ms = 20; c->limit_db = -3;
    c->def.set = 1; c->def.bypass = 1;
}

static int cfg_load(const char *path, config_t *c) {
    FILE *f = fopen(path, "r");
    if (!f) { logmsg("config %s: %s", path, strerror(errno)); return -1; }
    cfg_defaults(c);
    char line[2048], err[256]; int ln = 0, bad = 0;
    while (fgets(line, sizeof line, f)) {
        ln++;
        int inq = 0;   // strip comments: '#' at line start or after whitespace, outside quotes
        for (char *p = line; *p; p++) {
            if (*p == '"') inq = !inq;
            else if (*p == '#' && !inq && (p == line || isspace((unsigned char)p[-1]))) { *p = 0; break; }
        }
        char *s = trim(line);
        if (!*s) continue;
        char *eq = strchr(s, '=');
        if (!eq) { logmsg("config %s:%d: missing '='", path, ln); bad++; continue; }
        *eq = 0;
        char *k = trim(s), *v = trim(eq + 1);
        err[0] = 0;
        int p;
        if (!strcmp(k, "irdir")) snprintf(c->irdir, sizeof c->irdir, "%s", v);
        else if (!strcmp(k, "norm")) {
            if (!strcmp(v, "peak")) c->norm = 2; else if (!strcmp(v, "energy")) c->norm = 1; else if (!strcmp(v, "off")) c->norm = 0;
            else snprintf(err, sizeof err, "norm must be peak, energy or off");
        }
        else if (!strcmp(k, "ref")) c->ref_db = atof(v);
        else if (!strcmp(k, "maxlen")) c->maxlen_ms = atof(v);
        else if (!strcmp(k, "trim")) c->trim = !strcmp(v, "yes") || !strcmp(v, "on") || !strcmp(v, "1");
        else if (!strcmp(k, "warmup")) c->warmup_ms = atof(v);
        else if (!strcmp(k, "xfade")) c->xfade_ms = atof(v);
        else if (!strcmp(k, "limit")) c->limit_db = atof(v);
        else if (!strcmp(k, "directmon")) logmsg("config %s:%d: 'directmon' is no longer supported, ignored", path, ln);
        else if (!strcmp(k, "default")) parse_target(v, &c->def, err, sizeof err);
        else if (k[0] == 'I' && k[1] == 'R' && k[2] && !k[3] && slot_index(k[2]) >= 0) parse_target(v, &c->slot[slot_index(k[2])], err, sizeof err);
        else if ((p = parse_patch_key(k)) >= 0) parse_target(v, &c->patch[p], err, sizeof err);
        else snprintf(err, sizeof err, "unknown key '%s'", k);
        if (err[0]) { logmsg("config %s:%d: %s", path, ln, err); bad++; }
    }
    fclose(f);
    if (c->maxlen_ms < 10) c->maxlen_ms = 10;
    if (c->warmup_ms < 0) c->warmup_ms = 0;
    if (c->xfade_ms < 1) c->xfade_ms = 1;
    if (c->limit_db > 0) c->limit_db = 0;
    return bad ? -1 : 0;
}

// name token "IRx": at start or after a space, followed by a space or end of name
static int name_token(const char *name) {
    int n = strlen(name);
    for (int i = 0; i + 2 < n; i++)
        if (name[i] == 'I' && name[i + 1] == 'R' && (i == 0 || name[i - 1] == ' ') &&
            slot_index(name[i + 2]) >= 0 && (i + 3 == n || name[i + 3] == ' ')) return slot_index(name[i + 2]);
    return -1;
}

static const target_t *resolve(int patch, const char *name, char *why, size_t wl) {
    static const target_t bypass = { .set = 1, .bypass = 1 };
    char lb[8];
    int s = name ? name_token(name) : -1;
    why[0] = 0;
    if (s >= 0) {
        char sc = s < 10 ? '0' + s : s < 36 ? 'a' + s - 10 : 'A' + s - 36;
        if (cfg.slot[s].set) { snprintf(why, wl, "name token IR%c", sc); return &cfg.slot[s]; }
        snprintf(why, wl, "name token IR%c undefined, ", sc);
    }
    size_t l = strlen(why);
    if (patch >= 0 && patch < NPATCH && cfg.patch[patch].set) { snprintf(why + l, wl - l, "mapping %s", patch_label(patch, lb)); return &cfg.patch[patch]; }
    if (cfg.def.set) { snprintf(why + l, wl - l, "default"); return &cfg.def; }
    snprintf(why + l, wl - l, "bypass");
    return &bypass;
}

static void target_sig(const target_t *t, char *sig, size_t n) {
    if (t->bypass) snprintf(sig, n, "bypass|%g", t->gain_db);
    else {
        char full[1100]; resolve_path(t->path, full, sizeof full);
        snprintf(sig, n, "%s|%g|%d|%g|%g|%d", full, t->gain_db, cfg.norm, cfg.ref_db, cfg.maxlen_ms, cfg.trim);
    }
}

static void cfg_apply_rt(void) {
    atomic_store(&rt_limit, (float)pow(10, cfg.limit_db / 20));
    atomic_store(&rt_warmup, (long)(cfg.warmup_ms * RATE / 1000));
    atomic_store(&rt_xfade, (long)(cfg.xfade_ms * RATE / 1000));
}

// background preloading: loads at most one not yet cached IR per call; returns 0 when done
static int preload_idx;
static int cfg_preload_step(void) {
    char full[1100], err[1200];
    const int total = 1 + NPATCH + NSLOT;
    for (; preload_idx < total; preload_idx++) {
        int i = preload_idx;
        const target_t *t = i == 0 ? &cfg.def : i <= NPATCH ? &cfg.patch[i - 1] : &cfg.slot[i - 1 - NPATCH];
        if (!t->set || t->bypass) continue;
        resolve_path(t->path, full, sizeof full);
        int cached = 0;
        struct stat st;
        if (!stat(full, &st)) for (ircache_t *c = ircache; c; c = c->next) if (!strcmp(c->path, full) && c->mtime == st.st_mtime) { cached = 1; break; }
        if (cached) continue;
        if (!ir_get(full, err, sizeof err)) logmsg("config: %s", err);
        preload_idx++;
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------- setup
static void setup_global(void) {
    for (int c = 0; c < 3; c++) inring[c] = calloc(RING, sizeof(float));
    fftwf_import_wisdom_from_filename(opt.wisdom);
    double t0 = now();
    for (int si = 0; si < MAXSEG; si++) {
        int N = 2 * seg_B[si];
        float *a = fftwf_malloc(sizeof(float) * N), *b = fftwf_malloc(sizeof(float) * N);
        fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * (N / 2 + 2));
        plan_fwd[si] = fftwf_plan_dft_r2c_1d(N, a, X, FFTW_MEASURE);
        plan_inv[si] = fftwf_plan_dft_c2r_1d(N, X, b, FFTW_MEASURE);
    }
    if (now() - t0 > 1) { fftwf_export_wisdom_to_filename(opt.wisdom); logmsg("FFT planning took %.1f s (cached in %s)", now() - t0, opt.wisdom); }
}

// ---------------------------------------------------------------- selftest
// Runs two engines through the real code path (waiting for workers): engine A from the start,
// engine B activated mid-stream (tests history priming). Both are compared with brute force.
static int selftest(void) {
    const int P = opt.period;
    char err[1200];
    target_t t = { .set = 1 }; snprintf(t.path, sizeof t.path, "%s", opt.ir); t.gain_db = opt.gain_db;
    engine_t *a = engine_build(&t, err, sizeof err), *b = NULL;
    if (!a) { logmsg("%s", err); return 1; }
    logmsg("selftest IR: %s", a->desc);
    long n1 = 3 * 8192 + 5 * P;                       // not aligned to any block size
    long check_b = n1 + 16384 + 8192;                 // all segment gaps closed after this
    long len = check_b + a->irlen + 8192; len -= len % P;
    float *xin = malloc(sizeof(float) * len * 2);
    srand(1);
    for (long i = 0; i < len * 2; i++) xin[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    float *ya = calloc(len * 2, sizeof(float)), *yb = calloc(len * 2, sizeof(float));
    float x[3][MAXP], y[2][MAXP];
    engine_activate(a, 0);
    double t0 = now();
    for (long n = 0; n < len; n += P) {
        for (int i = 0; i < P; i++) { x[0][i] = xin[(n + i) * 2]; x[1][i] = xin[(n + i) * 2 + 1]; x[2][i] = 0.5f * (x[0][i] + x[1][i]); }
        input_store(x, P, n);
        if (n == n1) { b = engine_build(&t, err, sizeof err); engine_activate(b, n); }
        engine_post(a, n + P); if (b) engine_post(b, n + P);
        engine_process(a, x, P, n, 1, y);
        for (int i = 0; i < P; i++) for (int c = 0; c < CH; c++) ya[(n + i) * 2 + c] = y[a->NL == 2 ? c : 0][i];
        if (b) {
            engine_process(b, x, P, n, 1, y);
            for (int i = 0; i < P; i++) for (int c = 0; c < CH; c++) yb[(n + i) * 2 + c] = y[b->NL == 2 ? c : 0][i];
        }
    }
    double t1 = now();
    int rc = 0;
    for (int which = 0; which < 2; which++) {
        engine_t *e = which ? b : a; float *yo = which ? yb : ya;
        double err2 = 0, ref = 0;
        for (long i = which ? check_b : 0; i < len; i += 13) {
            for (int c = 0; c < CH; c++) {
                int ln = e->NL == 2 ? c : 0; double acc = 0;
                for (long tap = 0; tap < e->irlen && tap <= i; tap++) {
                    float xl = xin[(i - tap) * 2], xr = xin[(i - tap) * 2 + 1];
                    float v = e->lane_src[ln] == 0 ? xl : e->lane_src[ln] == 1 ? xr : 0.5f * (xl + xr);
                    acc += (double)e->irs[ln][tap] * v;
                }
                double d = yo[i * 2 + c] - acc;
                err2 += d * d; ref += acc * acc;
            }
        }
        double snr = 10 * log10(ref / (err2 + 1e-30));
        logmsg("selftest engine %c (%s): error %.1f dB below signal -> %s", which ? 'B' : 'A',
               which ? "activated mid-stream" : "from start", snr, snr > 80 ? "PASS" : "FAIL");
        if (snr <= 80) rc = 1;
    }
    logmsg("selftest: %ld samples (%.2f s audio) processed in %.2f s", len, (double)len / RATE, t1 - t0);
    engine_destroy(a); engine_destroy(b);
    return rc;
}

// ---------------------------------------------------------------- main / control loop
static void on_sig(int s) { (void)s; atomic_store(&running, 0); }

static void usage(const char *p) {
    fprintf(stderr,
        "usage: %s -C gt1conv.conf [options]      (patch mapping via config, see gt1conv.conf)\n"
        "       %s -i ir.wav [-g dB] [options]    (single IR, no MIDI)\n"
        "  -C FILE   config file (reloaded automatically when changed)\n"
        "  -i FILE   single IR file (instead of -C)\n"
        "  -g DB     gain for -i (dB)\n"
        "  -d DEV    ALSA device (default hw:GT1; MIDI uses DEV,0,0)\n"
        "  -p N      period size in frames, power of 2, 8..128 (default 8)\n"
        "  -f N      playback fill / queue in frames (default 16)\n"
        "  -m        mono: convolve (L+R)/2 once (half CPU); stereo IR -> L/R\n"
        "  -s SEC    statistics interval (default 60, 0 = off)\n"
        "  -c A,B,C,D  CPUs for audio,seg0,seg1,seg2 (default 3,2,1,1)\n"
        "  -S        selftest with -i: compare engine output with brute-force convolution\n", p, p);
}

int main(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "C:i:g:d:p:f:ms:c:Sh")) != -1) switch (c) {
        case 'C': opt.conf = optarg; break;
        case 'i': opt.ir = optarg; break;
        case 'g': opt.gain_db = atof(optarg); break;
        case 'd': opt.dev = optarg; break;
        case 'p': opt.period = atoi(optarg); break;
        case 'f': opt.fill = atoi(optarg); break;
        case 'm': opt.mono = 1; break;
        case 's': opt.stats = atoi(optarg); break;
        case 'c': sscanf(optarg, "%d,%d,%d,%d", &opt.cpu_audio, &opt.cpu_seg[0], &opt.cpu_seg[1], &opt.cpu_seg[2]); break;
        case 'S': opt.selftest = 1; break;
        default: usage(argv[0]); return c == 'h' ? 0 : 1;
    }
    if (!opt.conf && !opt.ir) { usage(argv[0]); return 1; }
    if (opt.period < 8 || opt.period > MAXP || (opt.period & (opt.period - 1))) { logmsg("period must be a power of 2 in 8..%d", MAXP); return 1; }
    if (opt.fill < opt.period) opt.fill = opt.period;

    struct stat cst = {0};
    if (opt.conf && !opt.selftest) {
        if (cfg_load(opt.conf, &cfg) < 0) { logmsg("config has errors, not starting"); return 1; }
        stat(opt.conf, &cst);
    } else {
        if (opt.conf) cfg_load(opt.conf, &cfg); else cfg_defaults(&cfg);   // selftest: use norm settings from config
        cfg.def.bypass = 0; snprintf(cfg.def.path, sizeof cfg.def.path, "%s", opt.ir ? opt.ir : ""); cfg.def.gain_db = opt.gain_db;
        opt.conf = opt.selftest ? opt.conf : NULL;
    }
    cfg_apply_rt();
    setup_global();
    if (opt.selftest) {
        if (!opt.ir) { logmsg("-S needs -i"); return 1; }
        return selftest();
    }

    struct sigaction sa = { .sa_handler = on_sig };
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);
    if (mlockall(MCL_CURRENT | MCL_FUTURE)) perror("mlockall (continuing)");

    {   // non-realtime work (control loop, IR loading, MIDI) stays on CPU 0, away from the audio cores
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    pthread_t at, mt;
    pthread_create(&at, NULL, audio_thread, NULL);
    if (opt.conf) pthread_create(&mt, NULL, midi_thread, NULL);

    char cur_sig[1400] = "", why[128], lb[8];
    engine_t *active = NULL;
    double last_stats = now(), last_cfg_check = now(), preload_t0 = now();
    int preloaded = 0;
    while (atomic_load(&running)) {
        usleep(20000);
        for (int i = 0; i < NRETIRE; i++) { engine_t *e = atomic_exchange(&retired_q[i], NULL); if (e) engine_destroy(e); }

        if (opt.conf && now() - last_cfg_check > 1.0) {
            last_cfg_check = now();
            struct stat st;
            if (!stat(opt.conf, &st) && (st.st_mtime != cst.st_mtime || st.st_size != cst.st_size)) {
                cst = st;
                static config_t nc;
                if (cfg_load(opt.conf, &nc) == 0) {
                    cfg = nc; cfg_apply_rt(); preload_idx = 0; preloaded = 0; cur_sig[0] = 0;
                    logmsg("config reloaded");
                } else logmsg("config has errors, keeping previous configuration");
            }
        }

        pthread_mutex_lock(&gt.mu);
        int patch = gt.patch, name_ok = gt.name_ok, conn = gt.connected;
        char name[17]; strcpy(name, gt.name);
        double changed_at = gt.changed_at;
        pthread_mutex_unlock(&gt.mu);
        // right after a patch change wait (briefly) for its name, so a name token does not cause a double switch
        if (!(conn && !name_ok && now() - changed_at < 0.3)) {
            const target_t *t = resolve(patch, name_ok ? name : NULL, why, sizeof why);
            char sig[1400]; target_sig(t, sig, sizeof sig);
            if (strcmp(sig, cur_sig) && !atomic_load(&pending) && !atomic_load(&rt_switching)) {
                char err[1200];
                engine_t *e = engine_build(t, err, sizeof err);
                if (!e) {
                    logmsg("error: %s", err);
                    if (!active) { target_t bp = { .set = 1, .bypass = 1 }; e = engine_build(&bp, err, sizeof err); }
                }
                if (e) { atomic_store(&pending, e); active = e; }
                snprintf(cur_sig, sizeof cur_sig, "%s", sig);
                char nm[17]; snprintf(nm, sizeof nm, "%s", name_ok ? name : "");
                logmsg("patch %s '%s' -> %s [%s]", patch_label(patch, lb), trim(nm), e ? e->desc : "(unchanged)", why);
            }
        }

        // load the remaining IRs in the background, one per loop, while no switch is in progress
        if (active && !preloaded && !atomic_load(&pending) && !atomic_load(&rt_switching)) {
            if (!cfg_preload_step()) { preloaded = 1; if (opt.conf) logmsg("all IRs loaded (%.1f s)", now() - preload_t0); }
        }
        if (preloaded == 0 && preload_idx == 0) preload_t0 = now();

        if (opt.stats > 0 && now() - last_stats >= opt.stats) {
            last_stats = now();
            char segs[160] = ""; int off = 0;
            engine_t *e = active;   // only destroyed after a newer engine became active
            if (e && !e->bypass)
                for (int si = 0; si < MAXSEG; si++) if (e->seg[si].active)
                    off += snprintf(segs + off, sizeof segs - off, " seg%d %ldus/%d", si,
                                    atomic_exchange(&e->seg[si].work_ns_max, 0) / 1000, atomic_exchange(&e->seg[si].miss, 0));
            unsigned pi = atomic_exchange(&st_pk_in, 0), po = atomic_exchange(&st_pk_out, 0);
            long dmin = atomic_exchange(&st_dmin, LONG_MAX), dmax = atomic_exchange(&st_dmax, 0);
            logmsg("stats: in %.1f dBFS, out %.1f dBFS, limited %d, xrun %d, audio %ldus, delay %ld..%ld,%s",
                   20 * log10(pi / 1073741824.0 + 1e-9), 20 * log10(po / 1073741824.0 + 1e-9),
                   atomic_exchange(&st_lim, 0), atomic_exchange(&st_xrun, 0), atomic_exchange(&st_proc_max, 0) / 1000,
                   dmin == LONG_MAX ? 0 : dmin, dmax, segs[0] ? segs : " bypass");
        }
    }
    atomic_store(&running, 0);
    pthread_join(at, NULL);
    if (opt.conf) pthread_join(mt, NULL);
    for (int i = 0; i < NRETIRE; i++) engine_destroy(atomic_exchange(&retired_q[i], NULL));
    engine_destroy(rt_nxt);
    engine_destroy(rt_cur);
    engine_destroy(atomic_exchange(&pending, NULL));
    logmsg("stopped");
    return 0;
}
