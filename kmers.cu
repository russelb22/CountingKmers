#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>         // for uint8_t, etc.
#include <cuda_runtime.h> 

static uint8_t lookup[85] = { 0 };

__constant__ uint8_t encode_lookup_d[256];
__constant__ char    decode_lookup_d[4];

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

void initialize_lookup()
{
    lookup['A'] = 0;
    lookup['C'] = 1;
    lookup['G'] = 2;
    lookup['T'] = 3;
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

int main() {
    // === Choose one of these FASTA files by uncommenting ===
    char filename[256] = { "GCA_900098775.1_35KB.fasta" };
    // char filename[256] = { "GCA_036705545.1._200MB.fasta" };
    // char filename[256] = { "GCA_000206225.1_31MB.fasta" };

    int k = 31; // SUPPORTED VALUES FOR k = 21 or 31
    printf("k = %d\n", k);
    printf("Filename: %s\n", filename);

    // Read the entire sequence from the FASTA (header lines stripped)
    char* sequence = read_fasta(filename);
    if (!sequence) {
        fprintf(stderr, "Failed to read FASTA file '%s'\n", filename);
        return EXIT_FAILURE;
    }

    // For a sequence of length N, you get N–k+1 k-mers; here we simply report N
    int num_kmers = strlen(sequence);
    printf("Number of k-mers: %d\n", num_kmers);

    free(sequence);

    initialize_lookup();
    initialize_lookup_d();

    return EXIT_SUCCESS;
