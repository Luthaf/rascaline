#ifndef RASCALINE_TORCH_OPENMP_HPP
#define RASCALINE_TORCH_OPENMP_HPP

#include <vector>

#include <ATen/Tensor.h>

#ifdef _OPENMP
#include <omp.h>
#else

int omp_get_num_threads();
int omp_get_thread_num();

#endif

namespace rascaline_torch {

class ThreadLocalTensor {
public:
    /// Zero-initialize all the tensors with the given options
    void init(int n_threads, at::IntArrayRef size, at::TensorOptions options = {});

    /// Get the tensor for the current thread
    at::Tensor get();

    /// Sum all the thread local tensors and return the result
    at::Tensor sum();

private:
    std::vector<at::Tensor> tensors_;
};


}

#endif
