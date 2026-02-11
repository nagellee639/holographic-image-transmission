/*
 * Holographic Image Transmission Protocol — C Implementation (v3)
 * ================================================================
 *
 * Packed-bit approach: never expands random bits into floats.
 *  - Dot product: quantise image to int16, XOR packed bits with
 *    sign-bit packing of image, use popcount → O(N/64) per measurement.
 *  - Accumulation: walk the bits and add/subtract measurement.
 *
 * Build:
 *   gcc -O3 -march=native -o holo_protocol holo_protocol.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -----------------------------------------------------------------------
 * xoshiro256** PRNG
 * ----------------------------------------------------------------------- */

typedef struct {
  uint64_t s[4];
} rng_t;

static inline uint64_t rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_t *r) {
  uint64_t *s = r->s;
  const uint64_t result = rotl(s[1] * 5, 7) * 9;
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

static void rng_seed(rng_t *r, uint64_t seed) {
  uint64_t z;
  for (int i = 0; i < 4; i++) {
    seed += 0x9e3779b97f4a7c15ULL;
    z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s[i] = z ^ (z >> 31);
  }
}

/* -----------------------------------------------------------------------
 * Box-Muller Gaussian
 * ----------------------------------------------------------------------- */

static double uniform01(rng_t *r) {
  return (double)(rng_next(r) >> 11) / (double)(1ULL << 53);
}

static double rand_normal(rng_t *r) {
  double u1 = uniform01(r), u2 = uniform01(r);
  if (u1 < 1e-15)
    u1 = 1e-15;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* -----------------------------------------------------------------------
 * Popcount helper (GCC/Clang builtin → compiles to hardware POPCNT)
 * ----------------------------------------------------------------------- */

static inline int popcnt64(uint64_t x) { return __builtin_popcountll(x); }

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

int main(int argc, char **argv) {
  if (argc != 9) {
    fprintf(stderr,
            "Usage: %s <in.raw> <W> <H> <N_meas> <seed> <sync> "
            "<snr_db|-1> <out.raw>\n",
            argv[0]);
    return 1;
  }

  const char *in_path = argv[1];
  int W = atoi(argv[2]);
  int H = atoi(argv[3]);
  int N = atoi(argv[4]);
  uint64_t seed = (uint64_t)atoll(argv[5]);
  int sync_int = atoi(argv[6]);
  double snr_db = atof(argv[7]);
  const char *out_path = argv[8];
  int noiseless = (snr_db < 0);
  int npix = W * H;

  /* Number of uint64 words needed to pack npix bits */
  int nwords = (npix + 63) / 64;

  /* --- Load image --- */
  uint8_t *img_u8 = malloc(npix);
  FILE *f = fopen(in_path, "rb");
  if (!f) {
    perror(in_path);
    return 1;
  }
  fread(img_u8, 1, npix, f);
  fclose(f);

  float *image = malloc(npix * sizeof(float));
  for (int i = 0; i < npix; i++)
    image[i] = (float)img_u8[i];
  free(img_u8);

  /* Accumulator */
  double *accum = calloc(npix, sizeof(double));

  /* Random bits buffer (one mask worth) */
  uint64_t *bits = malloc(nwords * sizeof(uint64_t));

  /* --- Noise setup --- */
  double noise_std = 0.0;
  rng_t noise_rng;
  rng_seed(&noise_rng, seed ^ 0xDEADBEEFULL);

  if (!noiseless) {
    /* Estimate signal power from 200 measurements */
    rng_t est;
    rng_seed(&est, seed);
    double pwr = 0;
    int ne = N < 200 ? N : 200;
    for (int m = 0; m < ne; m++) {
      for (int w = 0; w < nwords; w++)
        bits[w] = rng_next(&est);
      /* Compute dot product using the mask bits */
      double dot = 0;
      for (int i = 0; i < npix; i++) {
        int w = i >> 6; /* i / 64 */
        int b = i & 63; /* i % 64 */
        double sign = (bits[w] >> b) & 1 ? 1.0 : -1.0;
        dot += sign * image[i];
      }
      pwr += dot * dot;
    }
    pwr /= ne;
    noise_std = sqrt(pwr / pow(10.0, snr_db / 10.0));
  }

  /* --- Main loop --- */
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  rng_t rng;
  int epoch = 0, idx = 0;
  rng_seed(&rng, seed + (uint64_t)epoch * 997ULL);

  for (int m = 0; m < N; m++) {
    /* Generate packed mask bits */
    for (int w = 0; w < nwords; w++)
      bits[w] = rng_next(&rng);

    /* --- Encode: dot(mask, image) ---
     * mask[i] = +1 if bit is 1, -1 if bit is 0
     * dot = sum_i mask[i]*image[i]
     *     = sum(image where bit=1) - sum(image where bit=0)
     *     = 2 * sum(image where bit=1) - sum(image)
     *
     * We compute sum(image where bit=1) by walking 64 pixels
     * at a time.  For each uint64 of bits, we iterate set bits.
     * But even simpler: just iterate pixels.
     */
    double dot = 0.0;
    int i = 0;
    for (int w = 0; w < nwords && i < npix; w++) {
      uint64_t word = bits[w];
      int end = i + 64;
      if (end > npix)
        end = npix;
      while (i < end) {
        /* branchless: sign = 1 or -1 based on lowest bit */
        double sign = (double)((int)(word & 1) * 2 - 1);
        dot += sign * image[i];
        word >>= 1;
        i++;
      }
    }

    /* Channel noise */
    double measurement = dot;
    if (!noiseless)
      measurement += noise_std * rand_normal(&noise_rng);

    /* --- Decode: accum[i] += measurement * mask[i] --- */
    i = 0;
    for (int w = 0; w < nwords && i < npix; w++) {
      uint64_t word = bits[w];
      int end = i + 64;
      if (end > npix)
        end = npix;
      while (i < end) {
        double sign = (double)((int)(word & 1) * 2 - 1);
        accum[i] += measurement * sign;
        word >>= 1;
        i++;
      }
    }

    /* Sync epoch */
    idx++;
    if (idx >= sync_int) {
      epoch++;
      idx = 0;
      rng_seed(&rng, seed + (uint64_t)epoch * 997ULL);
    }

    if (m > 0 && m % 50000 == 0) {
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
      fprintf(stderr, "  %d / %d  (%.1f s, %.0f meas/s)\n", m, N, dt, m / dt);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  fprintf(stderr, "  Done: %d measurements in %.2f s  (%.0f meas/s)\n", N,
          elapsed, N / elapsed);

  /* --- Normalise → uint8 --- */
  double mn = accum[0], mx = accum[0];
  for (int i = 1; i < npix; i++) {
    if (accum[i] < mn)
      mn = accum[i];
    if (accum[i] > mx)
      mx = accum[i];
  }
  double range = mx - mn;
  if (range < 1e-12)
    range = 1.0;

  uint8_t *out = malloc(npix);
  for (int i = 0; i < npix; i++) {
    double v = (accum[i] - mn) / range * 255.0;
    out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5);
  }

  f = fopen(out_path, "wb");
  fwrite(out, 1, npix, f);
  fclose(f);
  fprintf(stderr, "  Output → %s\n", out_path);

  free(image);
  free(accum);
  free(bits);
  free(out);
  return 0;
}
