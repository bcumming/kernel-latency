#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_profile.h>
#include <roctracer_ext.h>

static void check_status(hipError_t status) {
    if(status != hipSuccess) {
        std::cerr << "error: HIP API call : "
                  << hipGetErrorString(status) << std::endl;
        exit(-1);
    }
}

template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = hipMalloc(&p, n*sizeof(T));
    check_status(status);
    return (T*)p;
}

template <typename T>
void free_device(T* p) {
    hipFree(p);
}

template <typename T>
T* malloc_host(size_t n) {
    return static_cast<T*>(malloc(n*sizeof(T)));
}

template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    auto status = hipMemcpy(to, from, n*sizeof(T), hipMemcpyHostToDevice);
    check_status(status);
}

template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    auto status = hipMemcpy(to, from, n*sizeof(T), hipMemcpyDeviceToHost);
    check_status(status);
}

static void device_synch() {
    hipDeviceSynchronize();
}

static void start_gpu_prof() {
    roctracer_start();
}

static void stop_gpu_prof() {
    roctracer_stop();
}
