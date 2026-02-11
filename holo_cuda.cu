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
 * Build (Library):
 *   nvcc -O3 -shared -Xcompiler -fPIC -o libholo.so holo_cuda.cu -lcurand
 * -lcublas -lm -DBUILD_LIB
 *
 * Build (CLI):
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

#define BATCH_SIZE 512 /* masks per batch — fits comfortably in P1000's 4GB */

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
 * C API for Python Integration (Shared Library)
 * ----------------------------------------------------------------------- */

struct TxState {
  int width, height, npix;
  uint64_t seed;
  float *d_image;
  float *d_masks;
  float *d_meas;
  curandGenerator_t curand_gen;
  cublasHandle_t cublas;
};

struct RxState {
  int width, height, npix;
  uint64_t seed;
  float *d_accum;
  float *d_masks;
  float *d_meas; /* device buffer for incoming measurements */
  curandGenerator_t curand_gen;
  cublasHandle_t cublas;
};

extern "C" {

TxState *tx_create(uint64_t seed, int width, int height, const float *h_image) {
  TxState *s = new TxState;
  s->seed = seed;
  s->width = width;
  s->height = height;
  s->npix = width * height;

  CHECK_CUBLAS(cublasCreate(&s->cublas));
  CHECK_CURAND(
      curandCreateGenerator(&s->curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(s->curand_gen, seed));

  CHECK_CUDA(cudaMalloc(&s->d_image, s->npix * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(s->d_image, h_image, s->npix * sizeof(float),
                        cudaMemcpyHostToDevice));

  size_t max_batch = BATCH_SIZE;
  CHECK_CUDA(cudaMalloc(&s->d_masks, max_batch * s->npix * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s->d_meas, max_batch * sizeof(float)));

  return s;
}

void tx_destroy(TxState *s) {
  if (!s)
    return;
  cudaFree(s->d_image);
  cudaFree(s->d_masks);
  cudaFree(s->d_meas);
  curandDestroyGenerator(s->curand_gen);
  cublasDestroy(s->cublas);
  delete s;
}

/* Generate 'count' measurements into 'h_output' */
void tx_generate(TxState *s, float *h_output, int count) {
  int batch_size = BATCH_SIZE;
  int processed = 0;

  float alpha = 1.0f, beta = 0.0f;

  while (processed < count) {
    int chunk = count - processed;
    if (chunk > batch_size)
      chunk = batch_size;

    CHECK_CURAND(curandGenerateUniform(s->curand_gen, s->d_masks,
                                       (size_t)chunk * s->npix));
    int threads = 256;
    int blocks = ((size_t)chunk * s->npix + threads - 1) / threads;
    float_to_mask<<<blocks, threads>>>(s->d_masks, chunk * s->npix);

    CHECK_CUBLAS(cublasSgemv(s->cublas, CUBLAS_OP_T, s->npix, chunk, &alpha,
                             s->d_masks, s->npix, s->d_image, 1, &beta,
                             s->d_meas, 1));

    CHECK_CUDA(cudaMemcpy(h_output + processed, s->d_meas,
                          chunk * sizeof(float), cudaMemcpyDeviceToHost));

    processed += chunk;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

RxState *rx_create(uint64_t seed, int width, int height) {
  RxState *s = new RxState;
  s->seed = seed;
  s->width = width;
  s->height = height;
  s->npix = width * height;

  CHECK_CUBLAS(cublasCreate(&s->cublas));
  CHECK_CURAND(
      curandCreateGenerator(&s->curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(s->curand_gen, seed));

  CHECK_CUDA(cudaMalloc(&s->d_accum, s->npix * sizeof(float)));
  CHECK_CUDA(cudaMemset(s->d_accum, 0, s->npix * sizeof(float)));

  size_t max_batch = BATCH_SIZE;
  CHECK_CUDA(cudaMalloc(&s->d_masks, max_batch * s->npix * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s->d_meas, max_batch * sizeof(float)));

  return s;
}

void rx_destroy(RxState *s) {
  if (!s)
    return;
  cudaFree(s->d_accum);
  cudaFree(s->d_masks);
  cudaFree(s->d_meas);
  curandDestroyGenerator(s->curand_gen);
  cublasDestroy(s->cublas);
  delete s;
}

void rx_accumulate(RxState *s, const float *h_measurements, int count) {
  int batch_size = BATCH_SIZE;
  int processed = 0;
  float alpha = 1.0f, beta = 1.0f;

  while (processed < count) {
    int chunk = count - processed;
    if (chunk > batch_size)
      chunk = batch_size;

    CHECK_CUDA(cudaMemcpy(s->d_meas, h_measurements + processed,
                          chunk * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CURAND(curandGenerateUniform(s->curand_gen, s->d_masks,
                                       (size_t)chunk * s->npix));
    int threads = 256;
    int blocks = ((size_t)chunk * s->npix + threads - 1) / threads;
    float_to_mask<<<blocks, threads>>>(s->d_masks, chunk * s->npix);

    CHECK_CUBLAS(cublasSgemv(s->cublas, CUBLAS_OP_N, s->npix, chunk, &alpha,
                             s->d_masks, s->npix, s->d_meas, 1, &beta,
                             s->d_accum, 1));

    processed += chunk;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Get current reconstruction (uint8) */
void rx_get_image(RxState *s, uint8_t *h_output, int total_measurements) {
  if (total_measurements <= 0)
    return;

  float *h_accum = (float *)malloc(s->npix * sizeof(float));
  CHECK_CUDA(cudaMemcpy(h_accum, s->d_accum, s->npix * sizeof(float),
                        cudaMemcpyDeviceToHost));

  /* Normalization: accum / N */
  for (int i = 0; i < s->npix; i++) {
    float v = h_accum[i] / (float)total_measurements;
    if (v < 0.0f)
      v = 0.0f;
    if (v > 255.0f)
      v = 255.0f;
    h_output[i] = (uint8_t)(v + 0.5f);
  }
  free(h_accum);
}

} /* extern "C" */

/* -----------------------------------------------------------------------
 * Main (CLI Simulation)
 * ----------------------------------------------------------------------- */
#ifndef BUILD_LIB
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

  float *d_image;
  float *d_accum;
  float *d_masks;
  float *d_meas;
  float *d_noise;

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

  float noise_std = 0.0f;
  if (!noiseless) {
    int n_est = (N < 200) ? N : 200;
    if (n_est > batch)
      n_est = batch;

    CHECK_CURAND(
        curandGenerateUniform(curand_gen, d_masks, (size_t)n_est * npix));
    int threads = 256;
    int blocks = ((size_t)n_est * npix + threads - 1) / threads;
    float_to_mask<<<blocks, threads>>>(d_masks, n_est * npix);

    float *h_meas_est = (float *)malloc(n_est * sizeof(float));
    float alpha = 1.0f, beta = 0.0f;
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

    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, seed));
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  int remaining = N;
  int epoch = 0;
  int idx_in_epoch = 0;
  int total_done = 0;

  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(
      curand_gen, seed + (unsigned long long)epoch * 997ULL));

  float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

  while (remaining > 0) {
    int left_in_epoch = sync_int - idx_in_epoch;
    int chunk = remaining < batch ? remaining : batch;
    if (chunk > left_in_epoch)
      chunk = left_in_epoch;

    CHECK_CURAND(
        curandGenerateUniform(curand_gen, d_masks, (size_t)chunk * npix));
    int threads = 256;
    int blocks_mask = ((size_t)chunk * npix + threads - 1) / threads;
    float_to_mask<<<blocks_mask, threads>>>(d_masks, chunk * npix);

    CHECK_CUBLAS(cublasSgemv(cublas, CUBLAS_OP_T, npix, chunk, &alpha, d_masks,
                             npix, d_image, 1, &beta_zero, d_meas, 1));

    if (!noiseless) {
      CHECK_CURAND(curandGenerateNormal(noise_gen, d_noise, (chunk + 1) & ~1,
                                        0.0f, noise_std));
      int blocks_n = (chunk + threads - 1) / threads;
      add_noise_kernel<<<blocks_n, threads>>>(d_meas, d_noise, 1.0f, chunk);
    }

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

  float *h_accum = (float *)malloc(npix * sizeof(float));
  CHECK_CUDA(cudaMemcpy(h_accum, d_accum, npix * sizeof(float),
                        cudaMemcpyDeviceToHost));
  uint8_t *out = (uint8_t *)malloc(npix);
  for (int i = 0; i < npix; i++) {
    float v = h_accum[i] / (float)N;
    if (v < 0.0f)
      v = 0.0f;
    if (v > 255.0f)
      v = 255.0f;
    out[i] = (uint8_t)(v + 0.5f);
  }

  f = fopen(out_path, "wb");
  fwrite(out, 1, npix, f);
  fclose(f);
  fprintf(stderr, "  Output → %s\n", out_path);

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
#endif
