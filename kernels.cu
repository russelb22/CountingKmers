// ========== kernels.cu ==========
#include <cuda_runtime.h>
#include <cstdint>

__global__
void build_edges_kernel(const uint64_t* encoded,
                        size_t           num_kmers,
                        int              k,
                        uint64_t*        prefixes,
                        uint64_t*        suffixes)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kmers) return;

    uint64_t v      = encoded[idx];
    // high (k-1)*2 bits
    prefixes[idx]   = v >> 2;
    // low (k-1)*2 bits
    uint64_t mask   = (1ULL << ((k-1)*2)) - 1;
    suffixes[idx]   = v & mask;
}

