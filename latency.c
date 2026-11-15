// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Sascha Ittner
//
// Round-trip latency test for a synchronous capture->playback loop.
//   default: playback hw:sndrpihifiberry, capture hw:GT1   (override with env PB= / CAP=)
// Emits a windowed chirp every 0.5 s on both output channels, records capture ch0
// into RAM, and analyses offline (cross-correlation) after the run so the audio loop
// is never blocked.
//
// usage: sudo [PB=hw:GT1] ./latency <period> <fill> <seconds>
#include <alsa/asoundlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>

#define RATE 44100
#define CH 2
static const char *cap_dev = "hw:GT1", *pb_dev = "hw:sndrpihifiberry";

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static snd_pcm_t *open_pcm(const char *dev, snd_pcm_stream_t st, snd_pcm_uframes_t period, unsigned nper, snd_pcm_uframes_t start_thr) {
    snd_pcm_t *h; int e;
    if ((e = snd_pcm_open(&h, dev, st, 0)) < 0) { fprintf(stderr, "open %s: %s\n", dev, snd_strerror(e)); exit(1); }
    snd_pcm_hw_params_t *hw; snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(h, hw);
    snd_pcm_hw_params_set_access(h, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(h, hw, SND_PCM_FORMAT_S32_LE);
    snd_pcm_hw_params_set_channels(h, hw, CH);
    snd_pcm_hw_params_set_rate(h, hw, RATE, 0);
    snd_pcm_hw_params_set_period_size(h, hw, period, 0);
    unsigned np = nper; snd_pcm_hw_params_set_periods_near(h, hw, &np, 0);
    if ((e = snd_pcm_hw_params(h, hw)) < 0) { fprintf(stderr, "hw_params %s: %s\n", dev, snd_strerror(e)); exit(1); }
    snd_pcm_uframes_t ps, bs; snd_pcm_hw_params_get_period_size(hw, &ps, 0); snd_pcm_hw_params_get_buffer_size(hw, &bs);
    snd_pcm_sw_params_t *sw; snd_pcm_sw_params_alloca(&sw);
    snd_pcm_sw_params_current(h, sw);
    snd_pcm_sw_params_set_avail_min(h, sw, period);
    snd_pcm_sw_params_set_start_threshold(h, sw, start_thr);
    snd_pcm_sw_params(h, sw);
    fprintf(stderr, "%s %s: period %lu buffer %lu\n", dev, st == SND_PCM_STREAM_PLAYBACK ? "pb" : "cap", ps, bs);
    return h;
}

int main(int argc, char **argv) {
    if (getenv("PB")) pb_dev = getenv("PB");
    if (getenv("CAP")) cap_dev = getenv("CAP");
    int period = argc > 1 ? atoi(argv[1]) : 64;
    int fill = argc > 2 ? atoi(argv[2]) : 128;     // playback prefill (frames) = output queue target
    int secs = argc > 3 ? atoi(argv[3]) : 10;
    struct sched_param sp = { .sched_priority = 80 };
    if (sched_setscheduler(0, SCHED_FIFO, &sp)) perror("sched_setscheduler (continuing)");

    // test signal: 1024-sample linear chirp 300..6000 Hz, Hann-tapered edges
    enum { TL = 1024 }; static float tmpl[TL];
    for (int i = 0; i < TL; i++) {
        double t = (double)i / RATE, T = (double)TL / RATE, f0 = 300, f1 = 6000;
        double ph = 2 * M_PI * (f0 * t + (f1 - f0) * t * t / (2 * T));
        double w = i < 64 ? 0.5 * (1 - cos(M_PI * i / 64)) : i > TL - 64 ? 0.5 * (1 - cos(M_PI * (TL - i) / 64)) : 1;
        tmpl[i] = w * sin(ph);
    }
    long total = (long)secs * RATE;
    int maxlag = 4096;
    float *rec = calloc(total + maxlag + TL + 8192, sizeof(float));
    int nclick_max = secs * 2 + 2; long *emit = calloc(nclick_max, sizeof(long)); int nemit = 0;
    mlockall(MCL_CURRENT | MCL_FUTURE);

    snd_pcm_t *cap = open_pcm(cap_dev, SND_PCM_STREAM_CAPTURE, period, 4, 1);
    snd_pcm_t *pb = open_pcm(pb_dev, SND_PCM_STREAM_PLAYBACK, period, 16, fill);

    int32_t *cbuf = calloc(period * CH, 4), *pbuf = calloc(period * CH, 4), *zero = calloc(8192 * CH, 4);
    long capf = 0, pbw = 0, next_click = RATE / 2; int click_pos = -1, xruns = 0;
    double S_t = 0, S_tt = 0, S_c = 0, S_ct = 0, S_p = 0, S_pt = 0, t0 = -1; long rn = 0;
    long dmin = 1 << 30, dmax = 0; double maxgap = 0, lastt = 0;

restart:
    snd_pcm_drop(pb); snd_pcm_drop(cap); snd_pcm_prepare(pb); snd_pcm_prepare(cap);
    snd_pcm_start(cap);
    snd_pcm_readi(cap, cbuf, period);                 // wait until the USB stream really delivers
    { snd_pcm_sframes_t a = snd_pcm_avail(cap); while (a >= period) { snd_pcm_readi(cap, cbuf, period); a -= period; } }
    pbw += snd_pcm_writei(pb, zero, fill);            // prefill -> starts playback
    click_pos = -1; next_click = capf + RATE / 2; lastt = 0;

    while (capf < total) {
        snd_pcm_sframes_t r = snd_pcm_readi(cap, cbuf, period);
        if (r < 0) { fprintf(stderr, "capture xrun at %ld: %s\n", capf, snd_strerror(r)); if (++xruns > 20) break; goto restart; }
        double tt = now();
        if (lastt > 0 && tt - lastt > maxgap) maxgap = tt - lastt;
        lastt = tt;
        for (int i = 0; i < r; i++) rec[capf + i] = cbuf[i * CH] / 2147483648.0f;

        // playback block (in the convolver this is where processing happens)
        for (int i = 0; i < period; i++) {
            long f = capf + i; float v = 0;
            if (click_pos < 0 && f == next_click && nemit < nclick_max) { click_pos = 0; emit[nemit++] = f; next_click += RATE / 2; }
            if (click_pos >= 0) { v = 0.5f * tmpl[click_pos]; if (++click_pos == TL) click_pos = -1; }
            int32_t s = (int32_t)(v * 2147483647.0f); pbuf[i * CH] = s; pbuf[i * CH + 1] = s;
        }
        capf += r;
        snd_pcm_sframes_t w = snd_pcm_writei(pb, pbuf, period);
        if (w < 0) { fprintf(stderr, "playback xrun at %ld: %s\n", capf, snd_strerror(w)); if (++xruns > 20) break; goto restart; }
        pbw += w;
        snd_pcm_sframes_t d, cd = 0;
        if (snd_pcm_delay(pb, &d) == 0) {
            snd_pcm_delay(cap, &cd);
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
            double t = tt; if (t0 < 0) t0 = t; t -= t0;
            double c = capf + cd, pp = pbw - d;
            S_t += t; S_tt += t * t; S_c += c; S_ct += c * t; S_p += pp; S_pt += pp * t; rn++;
        }
    }
    snd_pcm_drop(pb); snd_pcm_drop(cap);

    // ---- offline analysis ----
    // Emission at capture-frame index e means: the chirp sample was written to playback in the
    // same loop iteration in which capture frames [capf, capf+period) were read, at offset (e-capf).
    // So latency = index where it reappears in the capture stream - e.
    double sum = 0, sum2 = 0, mn = 1e9, mx = 0; int n = 0;
    for (int k = 0; k < nemit; k++) {
        long e = emit[k]; if (e + maxlag + TL > capf) break;
        int best = -1; double bv = 0, en = 0;
        for (int lag = 0; lag < maxlag; lag++) {
            const float *x = rec + e + lag; double c = 0;
            for (int j = 0; j < TL; j++) c += x[j] * tmpl[j];
            if (fabs(c) > bv) { bv = fabs(c); best = lag; }
        }
        for (int j = 0; j < TL; j++) en += (double)rec[e + best + j] * rec[e + best + j];
        double norm = bv / sqrt(en * (TL * 0.5) + 1e-20);   // ~1.0 = perfect match
        double pk = 0; for (int j = 0; j < TL; j++) if (fabs(rec[e + best + j]) > pk) pk = fabs(rec[e + best + j]);
        printf("latency %5d frames = %6.2f ms  (peak %.3f, match %.2f)\n", best, best * 1000.0 / RATE, pk, norm);
        if (norm > 0.5) { sum += best; sum2 += (double)best * best; n++; if (best < mn) mn = best; if (best > mx) mx = best; }
    }
    double den = rn * S_tt - S_t * S_t;
    double rc = (rn * S_ct - S_t * S_c) / den, rp = (rn * S_pt - S_t * S_p) / den;
    printf("rates vs CLOCK_MONOTONIC: capture %.3f Hz, playback %.3f Hz, diff %.1f ppm\n", rc, rp, (rp - rc) / rc * 1e6);
    printf("playback snd_pcm_delay min %ld max %ld, max loop gap %.2f ms\n", dmin, dmax, maxgap * 1e3);
    if (n) {
        double m = sum / n;
        printf("RESULT pb=%s period=%d fill=%d xruns=%d: mean %.1f frames (%.2f ms) min %.0f max %.0f sd %.2f, n=%d\n",
               pb_dev, period, fill, xruns, m, m * 1000 / RATE, mn, mx, sqrt(fmax(0, sum2 / n - m * m)), n);
    } else printf("RESULT pb=%s period=%d fill=%d xruns=%d: no valid detections\n", pb_dev, period, fill, xruns);
    return 0;
}
