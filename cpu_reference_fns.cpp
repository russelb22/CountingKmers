void encode_kmers_cpu(const char* seq, size_t seq_len, int k, 
                      std::vector<uint64_t>& out) {
  size_t num = seq_len >= k ? seq_len - k + 1 : 0;
  out.resize(num);
  for (size_t i = 0; i < num; ++i) {
    uint64_t v = 0;
    for (int j = 0; j < k; ++j) {
      uint8_t code = encode_lookup_h[(uint8_t)seq[i+j]];
      v = (v << 2) | code;
    }
    out[i] = v;
  }
}

void build_edges_cpu(const std::vector<uint64_t>& enc, int k,
                     std::vector<uint64_t>& pre,
                     std::vector<uint64_t>& suf) {
  size_t num = enc.size();
  pre.resize(num);
  suf.resize(num);
  uint64_t mask = (1ULL << ((k-1)*2)) - 1;
  for (size_t i = 0; i < num; ++i) {
    pre[i] = enc[i] >> 2;
    suf[i] = enc[i] & mask;
  }
}
