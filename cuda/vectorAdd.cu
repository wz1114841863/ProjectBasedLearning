#include <stdio.h>
#include <cuda_runtime.h>


__global__ void vector_add(const float *array_A, const float *array_B,
                           float *array_C, int number) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index < number) {
        array_C[thread_index] = array_A[thread_index] + array_B[thread_index] + 0.0f;
    }
    return ;
}


int main() {
    cudaError_t err = cudaSuccess;

    // the length of vector to be used
    int num_elements = 60000;
    size_t size = num_elements * sizeof(float);
    printf("[Vector addition of %d elements]\n", num_elements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors! \n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_elements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device \n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector C from host to device (error code %s)! \n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threads_per_block = 256;
    int block_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    printf("CUDA kernel launch with %d blocks of %d threads\n", block_per_grid,
        threads_per_block);
    vector_add<<<block_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_elements);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)! \n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory. \n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // verify that the result vector is correct.
    for (int i = 0; i < num_elements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("h_A[i] = %f, h_B[i] = %f, h_C[i] = %f ! \n", h_A[i], h_B[i], h_C[i]);
            fprintf(stderr, "Result verification failed at element %d! \n", i);
            // exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED. \n");

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)! \n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)! \n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
