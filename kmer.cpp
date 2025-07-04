#include <cstdio>
#include <cstring>
#include <cstdlib>

char* read_fasta(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    // Determine file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Determine chunk size based on file size
    size_t chunk_size;
    if (file_size <= (1 * 1024 * 1024)) {
        chunk_size = SMALL_FILE_CHUNK;
    }
    else if (file_size <= (50 * 1024 * 1024)) {
        chunk_size = MEDIUM_FILE_CHUNK;
    }
    else {
        chunk_size = LARGE_FILE_CHUNK;
    }

    // Allocate memory for filtered content
    char* filtered_content = (char*)malloc(file_size + 1); // Ensure enough space
    if (!filtered_content) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    size_t filtered_length = 0; // Track the length of filtered content
    char* buffer = (char*)malloc(chunk_size);
    if (!buffer) {
        perror("Memory allocation failed");
        free(filtered_content);
        fclose(file);
        return NULL;
    }

    // Read the file line by line
    while (fgets(buffer, chunk_size, file)) {
        if (buffer[0] == '>') {
            continue; // Skip header lines
        }

        // Remove newline characters and filter valid characters
        for (size_t i = 0; buffer[i] != '\0'; i++) {
            char c = buffer[i];
            if (c == 'A' || c == 'C' || c == 'T' || c == 'G') {
                filtered_content[filtered_length++] = c;
            }
        }
    }

    free(buffer);

    filtered_content[filtered_length] = '\0'; // Null-terminate
    fclose(file);

    if (filtered_length == 0) {
        fprintf(stderr, "Warning: No valid sequence data found.\n");
    }

    return filtered_content; // Return filtered content
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
    size_t num_kmers = strlen(sequence);
    printf("Number of k-mers: %zu million\n", num_kmers / (1000 * 1000));

    free(sequence);
    return EXIT_SUCCESS;
}
