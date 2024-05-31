#include <cassert>

#include "openmp.hpp"

#include <torch/torch.h>


using namespace rascaline_torch;

#ifndef _OPENMP

int omp_get_num_threads() {
    return 1;
}

int omp_get_thread_num() {
    return 0;
}

#endif

void ThreadLocalTensor::init(int n_threads, at::IntArrayRef size, at::TensorOptions options) {
    for (auto i=0; i<n_threads; i++) {
        tensors_.emplace_back(torch::zeros(size, options));
    }
}

at::Tensor ThreadLocalTensor::get() {
    return tensors_.at(omp_get_thread_num());
}

at::Tensor ThreadLocalTensor::sum() {
    assert(tensors_.size() > 0);
    auto sum = torch::zeros_like(tensors_[0]);
    for (const auto& tensor: tensors_) {
        sum += tensor;
    }
    return sum;
}
