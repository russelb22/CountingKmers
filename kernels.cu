// ========== kernels.cu ==========
#include <cuda_runtime.h>
#include <cstdint>

// extern __constant__ uint8_t encode_lookup_d[256];  // make sure this is initialized

__global__
void encode_kmers_kernel(const char* seq,
                         size_t    seq_len,
                         int       k,
                         uint64_t* encoded)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + k > seq_len) return;

    uint64_t val = 0;
    // pack k bases at 2 bits each
    for (int i = 0; i < k; ++i) {
        uint8_t code = encode_lookup_d[(uint8_t)seq[idx + i]];
        val = (val << 2) | code;
    }
    encoded[idx] = val;
}

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
