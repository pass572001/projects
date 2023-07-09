#include <stdio.h>

__global__ void reduce_min_max_sum_avg(int* input, int* output, int size) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input elements into shared memory
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]); // Min operation
            sdata[tid] = max(sdata[tid], sdata[tid + s]); // Max operation
            sdata[tid] += sdata[tid + s]; // Sum operation
        }
        __syncthreads();
    }

    // Write the reduced value to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int size = 1024; // Size of the input array
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Generate input data
    int* input = new int[size];
    for (int i = 0; i < size; ++i) {
        input[i] = i + 1;
    }

    // Allocate device memory
    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, num_blocks * sizeof(int));

    // Copy input data to device memory
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel for parallel reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_min_max_sum_avg<<<num_blocks, block_size, block_size * sizeof(int)>>>(d_input, d_output, size);
    cudaEventRecord(stop);

    // Allocate host memory for output
    int* output = new int[num_blocks];

    // Copy output data from device memory to host memory
    cudaMemcpy(output, d_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform reduction on host
    int min_val = output[0];
    int max_val = output[0];
    int sum_val = output[0];
    for (int i = 1; i < num_blocks; ++i) {
        min_val = min(min_val, output[i]); // Minimum value
        max_val = max(max_val, output[i]); // Maximum value
        sum_val += output[i]; // Sum value
    }
    float avg_val = static_cast<float>(sum_val) / size; // Average value

    printf("Min: %d\n", min_val);
    printf("Max: %d\n", max_val);
    printf("Sum: %d\n", sum_val);
    printf("Average: %.2f\n", avg_val);

    // Measure the execution time of sequential algorithm
    clock_t seq_start = clock();
    int seq_min = input[0];
    int seq_max = input[0];
    int seq_sum = input[0];
    for (int i = 1; i < size; ++i) {
        seq_min = min(seq_min, input[i]);
        seq_max = max(seq_max, input[i]);
        seq_sum += input[i];
    }
    float seq_avg = static_cast<float>(seq_sum) / size;
    clock_t seq_end = clock();
    float seq_time = static_cast<float>(seq_end - seq_start) / CLOCKS_PER_SEC * 1000;

    // Measure the execution time of parallel algorithm
    cudaEventSynchronize(stop);
    float par_time = 0;
    cudaEventElapsedTime(&par_time, start, stop);

    printf("Sequential Execution Time: %.4f ms\n", seq_time);
    printf("Parallel Execution Time: %.4f ms\n", par_time);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] input;
    delete[] output;

    return 0;
}

