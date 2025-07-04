// ========== host.cpp ==========
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// forward‐declared kernels (implement below)
__global__ void encode_kmers_kernel(const char* seq,
                                    size_t    seq_len,
                                    int       k,
                                    uint64_t* encoded);
__global__ void build_edges_kernel(const uint64_t* encoded,
                                   size_t           num_kmers,
                                   int              k,
                                   uint64_t*        prefixes,
                                   uint64_t*        suffixes);

int main() {
    // 1) Host sequence
    char* sequence = read_fasta("GCA_900098775.1_35KB.fasta");
    size_t seq_len = strlen(sequence);
    int    k       = 31;
    size_t num_kmers = seq_len >= k ? seq_len - k + 1 : 0;

    // 2) Copy sequence to device
    char*   d_seq;
    cudaMalloc(&d_seq, seq_len + 1);
    cudaMemcpy(d_seq, sequence, seq_len + 1, cudaMemcpyHostToDevice);

    // 3) Allocate device buffers for encoded k‐mers and edge lists
    uint64_t *d_encoded, *d_prefixes, *d_suffixes;
    cudaMalloc(&d_encoded,  num_kmers * sizeof(uint64_t));
    cudaMalloc(&d_prefixes, num_kmers * sizeof(uint64_t));
    cudaMalloc(&d_suffixes, num_kmers * sizeof(uint64_t));

    // 4) Launch encode_kmers_kernel
    int threads = 256;
    int blocks  = (num_kmers + threads - 1) / threads;
    encode_kmers_kernel<<<blocks,threads>>>(d_seq, seq_len, k, d_encoded);
    cudaDeviceSynchronize();

    // 5) Launch build_edges_kernel (prefix / suffix extraction)
    build_edges_kernel<<<blocks,threads>>>(d_encoded, num_kmers, k,
                                           d_prefixes, d_suffixes);
    cudaDeviceSynchronize();

    // 6) Use Thrust to build unique node set: concatenate prefixes & suffixes
    thrust::device_vector<uint64_t> nodes(2 * num_kmers);
    cudaMemcpy(thrust::raw_pointer_cast(nodes.data()),
               d_prefixes,
               num_kmers * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(nodes.data()) + num_kmers,
               d_suffixes,
               num_kmers * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    thrust::sort(nodes.begin(), nodes.end());
    auto new_end = thrust::unique(nodes.begin(), nodes.end());
    size_t num_nodes = new_end - nodes.begin();
    printf("Found %zu unique (k-1)-mers\n", num_nodes);

    // …next steps: map prefixes/suffixes → node indices, build CSR, compact unitigs…

    // cleanup
    cudaFree(d_seq);
    cudaFree(d_encoded);
    cudaFree(d_prefixes);
    cudaFree(d_suffixes);
    free(sequence);
    return 0;
}
