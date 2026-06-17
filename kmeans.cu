#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define N 1000000L
#define K 64
#define DIM 32
#define MAX_ITERS 100
#define TOL 1e-5

__device__ double device_dist_sq(const double *a, const double *b) {
    double sum = 0.0;

    for (int d = 0; d < DIM; d++) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }

    return sum;
}

__global__ void assign_points_kernel(const double *points, const double *centroids,
                                     int *assignments, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double *point = points + (size_t)i * DIM;
    int best = 0;
    double best_dist = device_dist_sq(point, centroids);

    for (int c = 1; c < K; c++) {
        double d = device_dist_sq(point, centroids + (size_t)c * DIM);
        if (d < best_dist) {
            best_dist = d;
            best = c;
        }
    }

    assignments[i] = best;
}

__global__ void reset_accumulators_kernel(double *sums, int *counts) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) {
        return;
    }

    counts[c] = 0;
    for (int d = 0; d < DIM; d++) {
        sums[(size_t)c * DIM + d] = 0.0;
    }
}

__global__ void accumulate_centroids_kernel(const double *points, const int *assignments,
                                            double *sums, int *counts, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int cluster = assignments[i];
    atomicAdd(&counts[cluster], 1);

    for (int d = 0; d < DIM; d++) {
        atomicAdd(&sums[(size_t)cluster * DIM + d], points[(size_t)i * DIM + d]);
    }
}

__global__ void finalize_centroids_kernel(const double *old_centroids, const double *sums,
                                          const int *counts, double *centroids, double *shift) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) {
        return;
    }

    if (counts[c] == 0) {
        for (int d = 0; d < DIM; d++) {
            centroids[(size_t)c * DIM + d] = old_centroids[(size_t)c * DIM + d];
        }
        return;
    }

    for (int d = 0; d < DIM; d++) {
        double updated = sums[(size_t)c * DIM + d] / (double)counts[c];
        double old = old_centroids[(size_t)c * DIM + d];
        centroids[(size_t)c * DIM + d] = updated;
        atomicAdd(shift, fabs(updated - old));
    }
}

__global__ void compute_inertia_kernel(const double *points, const double *centroids,
                                       const int *assignments, double *partial, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double *point = points + (size_t)i * DIM;
    const double *centroid = centroids + (size_t)assignments[i] * DIM;
    partial[i] = device_dist_sq(point, centroid);
}

static int check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static void generate_points(double *points) {
    srand(42);

    for (long i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            points[(size_t)i * DIM + d] = ((double)rand() / RAND_MAX) * 100.0;
        }
    }
}

static void init_centroids(const double *points, double *centroids) {
    for (int c = 0; c < K; c++) {
        for (int d = 0; d < DIM; d++) {
            centroids[(size_t)c * DIM + d] = points[(size_t)c * DIM + d];
        }
    }
}

int main(void) {
    const long n = N;
    const int block_size = 256;
    const int grid_points = (int)((n + block_size - 1) / block_size);
    const int grid_clusters = (K + block_size - 1) / block_size;

    size_t points_bytes = (size_t)n * DIM * sizeof(double);
    size_t centroids_bytes = (size_t)K * DIM * sizeof(double);
    size_t assignments_bytes = (size_t)n * sizeof(int);
    size_t sums_bytes = centroids_bytes;
    size_t counts_bytes = (size_t)K * sizeof(int);
    size_t partial_bytes = (size_t)n * sizeof(double);

    double *h_points = (double *)malloc(points_bytes);
    double *h_centroids = (double *)malloc(centroids_bytes);
    double *h_old_centroids = (double *)malloc(centroids_bytes);
    double *h_partial = (double *)malloc(partial_bytes);

    if (h_points == NULL || h_centroids == NULL || h_old_centroids == NULL ||
        h_partial == NULL) {
        fprintf(stderr, "host allocation failed\n");
        free(h_points);
        free(h_centroids);
        free(h_old_centroids);
        free(h_partial);
        return 1;
    }

    generate_points(h_points);
    init_centroids(h_points, h_centroids);

    double *d_points = NULL;
    double *d_centroids = NULL;
    double *d_old_centroids = NULL;
    double *d_sums = NULL;
    double *d_shift = NULL;
    double *d_partial = NULL;
    int *d_assignments = NULL;
    int *d_counts = NULL;

    if (check_cuda(cudaMalloc(&d_points, points_bytes), "cudaMalloc d_points") ||
        check_cuda(cudaMalloc(&d_centroids, centroids_bytes), "cudaMalloc d_centroids") ||
        check_cuda(cudaMalloc(&d_old_centroids, centroids_bytes), "cudaMalloc d_old_centroids") ||
        check_cuda(cudaMalloc(&d_sums, sums_bytes), "cudaMalloc d_sums") ||
        check_cuda(cudaMalloc(&d_shift, sizeof(double)), "cudaMalloc d_shift") ||
        check_cuda(cudaMalloc(&d_partial, partial_bytes), "cudaMalloc d_partial") ||
        check_cuda(cudaMalloc(&d_assignments, assignments_bytes), "cudaMalloc d_assignments") ||
        check_cuda(cudaMalloc(&d_counts, counts_bytes), "cudaMalloc d_counts")) {
        return 1;
    }

    if (check_cuda(cudaMemcpy(d_points, h_points, points_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy points H2D") ||
        check_cuda(cudaMemcpy(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy centroids H2D")) {
        return 1;
    }

    int iterations = 0;
    double h_shift = 0.0;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        assign_points_kernel<<<grid_points, block_size>>>(d_points, d_centroids, d_assignments, n);
        if (check_cuda(cudaGetLastError(), "assign_points_kernel")) {
            return 1;
        }

        if (check_cuda(cudaMemcpy(d_old_centroids, d_centroids, centroids_bytes,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy old centroids")) {
            return 1;
        }

        reset_accumulators_kernel<<<grid_clusters, block_size>>>(d_sums, d_counts);
        accumulate_centroids_kernel<<<grid_points, block_size>>>(d_points, d_assignments, d_sums,
                                                                 d_counts, n);

        h_shift = 0.0;
        if (check_cuda(cudaMemset(d_shift, 0, sizeof(double)), "cudaMemset d_shift") ||
            check_cuda(cudaGetLastError(), "centroid accumulation kernels")) {
            return 1;
        }

        finalize_centroids_kernel<<<grid_clusters, block_size>>>(d_old_centroids, d_sums, d_counts,
                                                                 d_centroids, d_shift);
        if (check_cuda(cudaGetLastError(), "finalize_centroids_kernel") ||
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
            return 1;
        }

        if (check_cuda(cudaMemcpy(&h_shift, d_shift, sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy shift D2H")) {
            return 1;
        }

        iterations = iter + 1;
        if (h_shift < TOL) {
            break;
        }
    }

    assign_points_kernel<<<grid_points, block_size>>>(d_points, d_centroids, d_assignments, n);
    compute_inertia_kernel<<<grid_points, block_size>>>(d_points, d_centroids, d_assignments,
                                                        d_partial, n);

    if (check_cuda(cudaGetLastError(), "final kernels") ||
        check_cuda(cudaDeviceSynchronize(), "final synchronize") ||
        check_cuda(cudaMemcpy(h_partial, d_partial, partial_bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy partial D2H")) {
        return 1;
    }

    double inertia = 0.0;
    for (long i = 0; i < n; i++) {
        inertia += h_partial[i];
    }

    printf("iterations = %d, inertia = %.6f\n", iterations, inertia);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_sums);
    cudaFree(d_shift);
    cudaFree(d_partial);
    cudaFree(d_assignments);
    cudaFree(d_counts);

    free(h_partial);
    free(h_old_centroids);
    free(h_centroids);
    free(h_points);
    return 0;
}
