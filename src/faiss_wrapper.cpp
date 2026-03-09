// faiss_wrapper.cpp
// Thin C extern "C" wrapper around FAISS IndexFlatL2, exposing a plain C API
// that Deno FFI can call via Deno.dlopen.

#include <faiss/IndexFlat.h>

extern "C" {
  faiss::IndexFlatL2* faiss_IndexFlatL2_new(int d) {
    return new faiss::IndexFlatL2(d);
  }
  
  void faiss_IndexFlatL2_add(faiss::IndexFlatL2* index, int64_t n, const float* x) {
    index->add(n, x);
  }

  void faiss_IndexFlatL2_search(faiss::IndexFlatL2* index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels) {
    index->search(n, x, k, distances, labels);
  }

  void faiss_IndexFlatL2_free(faiss::IndexFlatL2* index) {
    delete index;
  }
}
