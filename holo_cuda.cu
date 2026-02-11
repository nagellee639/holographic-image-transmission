/*
 * Holographic Image Transmission Protocol — CUDA Implementation
 * ==============================================================
 *
 * Uses cuRAND for GPU-parallel random number generation and
 * batched matrix operations for encode/decode.
 *
 * Strategy:
 *  - Keep image and accumulator on GPU
 *  - Process measurements in batches of BATCH_SIZE
 *  - For each batch:
 *    1. Generate BATCH_SIZE × num_pixels random ±1 masks via cuRAND
 *    2. Encode: dot products via cublasSgemv (masks × image)
 *    3. Add channel noise (on GPU)
 *    4. Decode: accumulate via cublasSgemv (masks^T × measurements)
 *
 * Build:
 *   nvcc -O3 -o holo_cuda holo_cuda.cu -lcurand -lcublas -lm
 *
 * Usage: same as the C version
 *   ./holo_cuda <in.raw> <W> <H> <N_meas> <seed> <sync> <snr|-1> <out.raw>
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 1000 /* masks per batch — fits comfortably in P1000's 4GB \
                         */
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CHECK_CURAND(call)                                                     \
  do {                                                                         \
    curandStatus_t e = (call);                                                 \
    if (e != CURAND_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuRAND error %s:%d: %d\n", __FILE__, __LINE__, e);      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t e = (call);                                                 \
    if (e != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, e);      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* -----------------------------------------------------------------------
 * Kernel: convert uniform random floats to ±1 mask values
 * cuRAND generates floats in (0,1]; we threshold at 0.5
 * ----------------------------------------------------------------------- */
__global__ void float_to_mask(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = (data[idx] > 0.5f) ? 1.0f : -1.0f;
  }
}

/* -----------------------------------------------------------------------
 * Kernel: add Gaussian noise to measurements (in-place)
 * noise is pre-generated as N(0,1), scaled by noise_std
 * ----------------------------------------------------------------------- */
__global__ void add_noise_kernel(float *measurements, const float *noise,
                                 float noise_std, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    measurements[idx] += noise_std * noise[idx];
  }
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
  if (argc != 9) {
    fprintf(stderr,
            "Usage: %s <in.raw> <W> <H> <N_meas> <seed> <sync> "
            "<snr|-1> <out.raw>\n",
            argv[0]);
    return 1;
  }

  const char *in_path = argv[1];
  int W = atoi(argv[2]);
  int H = atoi(argv[3]);
  int N = atoi(argv[4]);
  unsigned long long seed = (unsigned long long)atoll(argv[5]);
  int sync_int = atoi(argv[6]);
  double snr_db = atof(argv[7]);
  const char *out_path = argv[8];
  int noiseless = (snr_db < 0);
  int npix = W * H;

  /* --- Load image on host --- */
  uint8_t *img_u8 = (uint8_t *)malloc(npix);
  FILE *f = fopen(in_path, "rb");
  if (!f) {
    perror(in_path);
    return 1;
  }
  fread(img_u8, 1, npix, f);
  fclose(f);

  float *h_image = (float *)malloc(npix * sizeof(float));
  for (int i = 0; i < npix; i++)
    h_image[i] = (float)img_u8[i];
  free(img_u8);

  /* --- GPU setup --- */
  cublasHandle_t cublas;
  CHECK_CUBLAS(cublasCreate(&cublas));

  curandGenerator_t curand_gen, noise_gen;
  CHECK_CURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, seed));

  if (!noiseless) {
    CHECK_CURAND(curandCreateGenerator(&noise_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(
        curandSetPseudoRandomGeneratorSeed(noise_gen, seed ^ 0xDEADBEEF));
  }

  /* --- Device memory --- */
  float *d_image; /* (npix,)           — source image */
  float *d_accum; /* (npix,)           — accumulator (float32) */
  float *d_masks; /* (batch × npix)    — random masks */
  float *d_meas;  /* (batch,)          — measurements */
  float *d_noise; /* (batch,)          — noise samples */

  CHECK_CUDA(cudaMalloc(&d_image, npix * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_accum, npix * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_accum, 0, npix * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_image, h_image, npix * sizeof(float),
                        cudaMemcpyHostToDevice));

  int batch = BATCH_SIZE;
  CHECK_CUDA(cudaMalloc(&d_masks, (size_t)batch * npix * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_meas, batch * sizeof(float)));
  if (!noiseless)
    CHECK_CUDA(cudaMalloc(&d_noise, batch * sizeof(float)));

  /* --- Noise std estimate (quick host-side pass if needed) --- */
  float noise_std = 0.0f;
  if (!noiseless) {
    /* Generate a small batch of masks, compute dot products on GPU,
       pull back to host to estimate signal power */
    int n_est = (N < 200) ? N : 200;
    if (n_est > batch)
      n_est = batch;

    /* Generate masks */
    CHECK_CURAND(
        curandGenerateUniform(curand_gen, d_masks, (size_t)n_est * npix));
    int threads = 256;
    int blocks = ((size_t)n_est * npix + threads - 1) / threads;
    float_to_mask<<<blocks, threads>>>(d_masks, n_est * npix);

    /* dot products: meas = masks @ image  (each row dot image) */
    /* Using cublasSgemv in a loop for the estimation phase */
    float *h_meas_est = (float *)malloc(n_est * sizeof(float));
    float alpha = 1.0f, beta = 0.0f;
    /* Actually use Sgemv: masks is (n_est × npix), image is (npix × 1)
     * result is (n_est × 1)
     * cublas uses column-major, so masks in row-major = masks^T in col-major
     * We want y = masks * x, which in col-major is y = masks^T * x → use Trans
     */
    CHECK_CUBLAS(cublasSgemv(cublas, CUBLAS_OP_T, npix, n_est, &alpha, d_masks,
                             npix, d_image, 1, &beta, d_meas, 1));
    CHECK_CUDA(cudaMemcpy(h_meas_est, d_meas, n_est * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double pwr = 0;
    for (int i = 0; i < n_est; i++)
      pwr += (double)h_meas_est[i] * h_meas_est[i];
    pwr /= n_est;
    noise_std = (float)sqrt(pwr / pow(10.0, snr_db / 10.0));
    free(h_meas_est);

    /* Reset the generator so the actual run starts fresh */
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, seed));
  }

  /* --- Main loop: process in batches --- */
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  int remaining = N;
  int epoch = 0;
  int idx_in_epoch = 0;
  int total_done = 0;

  /* We re-seed cuRAND per sync epoch for determinism */
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(
      curand_gen, seed + (unsigned long long)epoch * 997ULL));

  float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

  while (remaining > 0) {
    /* How many measurements in this batch? */
    int left_in_epoch = sync_int - idx_in_epoch;
    int chunk = remaining < batch ? remaining : batch;
    if (chunk > left_in_epoch)
      chunk = left_in_epoch;

    /* 1. Generate random masks on GPU */
    CHECK_CURAND(
        curandGenerateUniform(curand_gen, d_masks, (size_t)chunk * npix));
    int threads = 256;
    int blocks_mask = ((size_t)chunk * npix + threads - 1) / threads;
    float_to_mask<<<blocks_mask, threads>>>(d_masks, chunk * npix);

    /* 2. Encode: measurements = masks × image
     * masks is row-major (chunk × npix), stored as column-major: (npix × chunk)
     * So: meas(chunk×1) = masks^T_colmaj(chunk×npix) × image(npix×1)
     *   → cublasSgemv(Trans, npix, chunk, ...)
     */
    CHECK_CUBLAS(cublasSgemv(cublas, CUBLAS_OP_T, npix, chunk, &alpha, d_masks,
                             npix, d_image, 1, &beta_zero, d_meas, 1));

    /* 3. Add channel noise */
    if (!noiseless) {
      CHECK_CURAND(curandGenerateNormal(noise_gen, d_noise,
                                        /* must be even */
                                        (chunk + 1) & ~1, 0.0f, noise_std));
      int blocks_n = (chunk + threads - 1) / threads;
      add_noise_kernel<<<blocks_n, threads>>>(d_meas, d_noise, 1.0f, chunk);
    }

    /* 4. Decode: accum += masks^T × measurements
     * accum(npix×1) += masks_colmaj(npix×chunk) × meas(chunk×1)
     *   → cublasSgemv(NoTrans, npix, chunk, ...)
     */
    CHECK_CUBLAS(cublasSgemv(cublas, CUBLAS_OP_N, npix, chunk, &alpha, d_masks,
                             npix, d_meas, 1, &beta_one, d_accum, 1));

    remaining -= chunk;
    total_done += chunk;
    idx_in_epoch += chunk;
    if (idx_in_epoch >= sync_int) {
      epoch++;
      idx_in_epoch = 0;
      CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(
          curand_gen, seed + (unsigned long long)epoch * 997ULL));
    }

    if (total_done % 10000 < chunk || remaining == 0) {
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
      fprintf(stderr, "  %d / %d  (%.1f s, %.0f meas/s)\n", total_done, N, dt,
              total_done / dt);
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  fprintf(stderr, "  Done: %d measurements in %.2f s  (%.0f meas/s)\n", N,
          elapsed, N / elapsed);

  /* --- Copy accumulator back, normalise --- */
  float *h_accum = (float *)malloc(npix * sizeof(float));
  CHECK_CUDA(cudaMemcpy(h_accum, d_accum, npix * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float mn = h_accum[0], mx = h_accum[0];
  for (int i = 1; i < npix; i++) {
    if (h_accum[i] < mn)
      mn = h_accum[i];
    if (h_accum[i] > mx)
      mx = h_accum[i];
  }
  float range = mx - mn;
  if (range < 1e-12f)
    range = 1.0f;

  uint8_t *out = (uint8_t *)malloc(npix);
  for (int i = 0; i < npix; i++) {
    float v = (h_accum[i] - mn) / range * 255.0f;
    out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
  }

  f = fopen(out_path, "wb");
  fwrite(out, 1, npix, f);
  fclose(f);
  fprintf(stderr, "  Output → %s\n", out_path);

  /* Cleanup */
  cudaFree(d_image);
  cudaFree(d_accum);
  cudaFree(d_masks);
  cudaFree(d_meas);
  if (!noiseless) {
    cudaFree(d_noise);
    curandDestroyGenerator(noise_gen);
  }
  curandDestroyGenerator(curand_gen);
  cublasDestroy(cublas);
  free(h_image);
  free(h_accum);
  free(out);
  return 0;
}
