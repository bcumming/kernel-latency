#include <iostream>

#if defined XXCUDA
    #include "util_cuda.h"
#elif defined XXHIP
    #include "util_cuda.h"
#else
    #error "one of XXCUDA or XXHIP must #defined"
#endif


// read command line arguments
int read_arg(int argc, char** argv, int index, int default_value) {
    if(argc>index) {
        try {
            auto n = std::stoi(argv[index]);
            if(n<0) {
                return default_value;
            }
            return n;
        }
        catch (std::exception e) {
            std::cout << "error : invalid argument \'" << argv[index]
                      << "\', expected a positive integer." << std::endl;
            exit(1);
        }
    }

    return default_value;
}

namespace kernels {

__global__
void empty(unsigned n) {}

__global__
void axpy(double *y, const double* x, double alpha, unsigned n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i<n) {
        y[i] += alpha*x[i];
    }
}

__device__
double f(double x) {
    return exp(cos(x))-2;
};

__device__
double fp(double x) {
    return -sin(x) * exp(cos(x));
};

__global__
void newton(double *x, unsigned n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) {
        auto x0 = x[i];
        for(int iter=0; iter<5; ++iter) {
            x0 -= f(x0)/fp(x0);
        }
        x[i] = x0;
    }
}

}

int main(int argc, char** argv) {
    const unsigned pow = read_arg(argc, argv, 1, 20);
    const unsigned n = 2 << pow;
    const unsigned block_dim = read_arg(argc, argv, 2, 128);
    const unsigned grid_dim = (n-1)/block_dim + 1;

    std::cout << "n " << n << " blockdim " << block_dim << " griddim " << grid_dim << "\n";

    // Run the newton kernel a bunch of times on a larger array to "warm up"
    {
        unsigned ni = 2<<24;
        double* xhi = malloc_host<double>(ni);
        double* xi = malloc_device<double>(ni);
        std::fill(xhi, xhi+ni, 2.3);
        copy_to_device<double>(xhi, xi, ni);
        for (auto i=0; i<1000; ++i) {
            kernels::newton<<<grid_dim, block_dim>>>(xi, ni);
        }
        std::free(xhi);
        free_device(xi);
    }


    double* xh = malloc_host<double>(n);
    double* yh = malloc_host<double>(n);
    std::fill(xh, xh+n, 2.0);
    std::fill(yh, yh+n, 1.0);

    double* x = malloc_device<double>(n);
    double* y = malloc_device<double>(n);
    copy_to_device<double>(xh, x, n);
    copy_to_device<double>(yh, y, n);

    device_synch();
    start_gpu_prof();

    int nruns = 10;

    for (auto i=0; i<nruns; ++i) {
        kernels::newton<<<grid_dim, block_dim>>>(x, n);
    }

    for (auto i=0; i<nruns; ++i) {
        kernels::axpy<<<grid_dim, block_dim>>>(y, x, 2.0, n);
    }

    for (auto i=0; i<nruns; ++i) {
        kernels::empty<<<grid_dim, block_dim>>>(n);
    }

    stop_gpu_prof();

    std::free(xh);
    std::free(yh);
    free_device(x);
    free_device(y);

    return 0;
}

