#include <cstdlib>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

static void check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    check_status(status);
    return static_cast<T*>(p);
}

template <typename T>
void free_device(T* p) {
    cudaFree(p);
}

template <typename T>
T* malloc_host(size_t n) {
    return static_cast<T*>(malloc(n*sizeof(T)));
}

template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyHostToDevice);
    check_status(status);
}

template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyDeviceToHost);
    check_status(status);
}

static void device_synch() {
    cudaDeviceSynchronize();
}

static void start_gpu_prof() {
    cudaProfilerStart();
}

static void stop_gpu_prof() {
    cudaProfilerStop();
}
