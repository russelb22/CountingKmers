// ========== host.cu ==========
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

// extern __constant__ uint8_t encode_lookup_d[256];  // make sure this is initialized
static uint8_t lookup[85] = { 0 };

__constant__ uint8_t encode_lookup_d[256];
__constant__ char    decode_lookup_d[4];

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

char* read_fasta(const char* fname) {
    FILE* f = fopen(fname, "r");
    if (!f) return nullptr;

    // Skip header lines and count sequence length
    int cap = 0, len = 0;
    char* buf = nullptr;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '>') continue;
        int l = strlen(line);
        // strip newline
        if (line[l-1] == '\n') line[--l] = '\0';
        if (len + l + 1 > cap) {
            cap = (len + l + 1) * 2;
            buf = (char*)realloc(buf, cap);
        }
        memcpy(buf + len, line, l);
        len += l;
    }
    if (buf) buf[len] = '\0';
    fclose(f);
    return buf;
}

void initialize_lookup_d()
{
    // 1) Prepare host‐side decode lookup and copy it
    const char decode_lookup_h[4] = { 'A', 'C', 'G', 'T' };
    cudaMemcpyToSymbol(decode_lookup_d,
                       decode_lookup_h,
                       sizeof(decode_lookup_h));


   // 2) Copy the entire table into device constant memory
    cudaMemcpyToSymbol(encode_lookup_d,
                       lookup,
                       sizeof(lookup));
}

int main()
{
    // 0) Host sequence
    char* sequence = read_fasta("GCA_900098775.1_35KB.fasta");
    size_t seq_len = strlen(sequence);
    int    k       = 21;
    size_t num_kmers = seq_len >= k ? seq_len - k + 1 : 0;

    // 1) build the host-side table
    uint8_t encode_lookup_h[256] = {0};
    encode_lookup_h[(uint8_t)'A'] = 0x0;
    encode_lookup_h[(uint8_t)'C'] = 0x1;
    encode_lookup_h[(uint8_t)'G'] = 0x2;
    encode_lookup_h[(uint8_t)'T'] = 0x3;

    // 2) copy it to the __constant__ array
    cudaMemcpyToSymbol(encode_lookup_d,
                      encode_lookup_h,
                      sizeof(encode_lookup_h),
                      /* offset */ 0,
                      cudaMemcpyHostToDevice);

    // (optionally) check for errors
    cudaDeviceSynchronize();

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
    int num_nodes = new_end - nodes.begin();
    printf("Found %d unique (k-1)-mers\n", num_nodes);

    // …next steps: map prefixes/suffixes → node indices, build CSR, compact unitigs…

    // cleanup
    cudaFree(d_seq);
    cudaFree(d_encoded);
    cudaFree(d_prefixes);
    cudaFree(d_suffixes);
    free(sequence);
    return 0;
}
