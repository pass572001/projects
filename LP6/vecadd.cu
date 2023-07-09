#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

// CUDA kernel for vector addition
__global__
void vectorAddition(const int* A, const int* B, int* C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] + B[index];
    }
}

// Function to perform vector addition sequentially
void sequentialVectorAddition(const int* A, const int* B, int* C, int size)
{
    for (int i = 0; i < size; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int size = 1000000;  // Size of the vectors

    // Allocate memory for vectors on host
    int* A = new int[size];
    int* B = new int[size];
    int* C = new int[size];

    // Initialize vectors with random values
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < size; ++i)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate memory for vectors on device
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * size);
    cudaMalloc((void**)&d_B, sizeof(int) * size);
    cudaMalloc((void**)&d_C, sizeof(int) * size);

    // Copy input vectors from host to device
    cudaMemcpy(d_A, A, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Set up thread configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Start timer for parallel algorithm
    auto startParallel = std::chrono::high_resolution_clock::now();

    // Launch kernel for vector addition in parallel
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // Wait for kernel to finish execution
    cudaDeviceSynchronize();

    // End timer for parallel algorithm
    auto endParallel = std::chrono::high_resolution_clock::now();

    // Copy result vector from device to host
    cudaMemcpy(C, d_C, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Start timer for sequential algorithm
    auto startSequential = std::chrono::high_resolution_clock::now();

    // Perform vector addition sequentially
    sequentialVectorAddition(A, B, C, size);

    // End timer for sequential algorithm
    auto endSequential = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time for parallel algorithm
    auto durationParallel = std::chrono::duration_cast<std::chrono::microseconds>(endParallel - startParallel);

    // Calculate elapsed time for sequential algorithm
    auto durationSequential = std::chrono::duration_cast<std::chrono::microseconds>(endSequential - startSequential);

    // Calculate speedup
    double speedup = static_cast<double>(durationSequential.count()) / durationParallel.count();

    // Print performance results
    std::cout << "Parallel Algorithm Time: " << durationParallel.count() << " microseconds" << std::endl;
    std::cout << "Sequential Algorithm Time: " << durationSequential.count() << " microseconds" << std::endl;
    std::cout << "Speedup: " << speedup << std::endl;

    // Print the results of vector addition (first 10 elements)
    std::cout << "Vector Addition Result:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

