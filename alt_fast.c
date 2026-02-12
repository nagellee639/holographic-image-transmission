/*
 * alt_fast.c — Optimized C implementations for alternative protocols.
 *
 * Compile:
 *   gcc -O3 -shared -fPIC -fopenmp -o libalt_fast.so alt_fast.c -lm
 *
 * All functions receive pre-computed random parameters from Python (numpy PRNG)
 * so that TX and RX stay perfectly synchronized.
 */

#include <math.h>
#include <stdlib.h>

/* ============================================================
 * Line Protocol — Bresenham line average
 * ============================================================ */

static float line_average(const float *img, int width, int height, int x0,
                          int y0, int x1, int y1) {
  double sum = 0;
  int count = 0;
  int dx = abs(x1 - x0);
  int dy = abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1;
  int sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;
  int loop_count = 0;
  int max_loops = dx + dy + 100; // Safety guard

  while (loop_count++ < max_loops) {
    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
      sum += img[y0 * width + x0];
      count++;
    }

    if (x0 == x1 && y0 == y1)
      break;

    int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0 += sy;
    }
  }

  return count > 0 ? (float)(sum / count) : 0.0f;
}

/* Generate line measurements in batch (OpenMP parallel) */
void line_tx_batch(const float *img, int width, int height, const int *x0s,
                   const int *y0s, const int *x1s, const int *y1s,
                   float *output, int count) {
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < count; i++) {
    output[i] =
        line_average(img, width, height, x0s[i], y0s[i], x1s[i], y1s[i]);
  }
}

/* Accumulate line measurements into image (sequential — shared accum) */
void line_rx_batch(double *accum, int *counts, int width, int height,
                   const int *x0s, const int *y0s, const int *x1s,
                   const int *y1s, const double *measurements, int count) {
  for (int i = 0; i < count; i++) {
    double val = measurements[i];
    int x0 = x0s[i], y0 = y0s[i], x1 = x1s[i], y1 = y1s[i];
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int loop_count = 0;
    int max_loops = dx + dy + 100; // Safety guard

    while (loop_count++ < max_loops) {
      if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        int idx = y0 * width + x0;
        accum[idx] += val;
        counts[idx]++;
      }

      if (x0 == x1 && y0 == y1)
        break;

      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }
  }
}

/* ============================================================
 * Block Protocol — Rectangle average
 * ============================================================ */

/* Generate block measurements in batch (OpenMP parallel) */
void block_tx_batch(const float *img, int width, int height, const int *bws,
                    const int *bhs, const int *x0s, const int *y0s,
                    float *output, int count) {
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < count; i++) {
    int bw = bws[i], bh = bhs[i];
    int x0 = x0s[i], y0 = y0s[i];
    double sum = 0;
    int n = bw * bh;
    for (int dy = 0; dy < bh; dy++) {
      for (int dx = 0; dx < bw; dx++) {
        sum += img[(y0 + dy) * width + (x0 + dx)];
      }
    }
    output[i] = (float)(sum / n);
  }
}

/* Accumulate block measurements (sequential — shared accum) */
void block_rx_batch(double *accum, int *counts, int width, int height,
                    const int *bws, const int *bhs, const int *x0s,
                    const int *y0s, const double *measurements, int count) {
  for (int i = 0; i < count; i++) {
    double val = measurements[i];
    int bw = bws[i], bh = bhs[i];
    int x0 = x0s[i], y0 = y0s[i];
    for (int dy = 0; dy < bh; dy++) {
      for (int dx = 0; dx < bw; dx++) {
        int idx = (y0 + dy) * width + (x0 + dx);
        accum[idx] += val;
        counts[idx]++;
      }
    }
  }
}
