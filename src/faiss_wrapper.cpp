// faiss_wrapper.cpp
// Thin C extern "C" wrapper around FAISS IndexFlatL2 and Clustering, exposing a plain C API
// that Deno FFI can call via Deno.dlopen.

#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>

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

  // Clustering wrapper functions

  faiss::Clustering* faiss_Clustering_new(int d, int k, int niter) {
    faiss::Clustering* clustering = new faiss::Clustering(d, k);
    clustering->niter = niter;
    return clustering;
  }

  int faiss_Clustering_train(faiss::Clustering* clustering, int n, const float* x) {
    faiss::IndexFlatL2 index(clustering->d);
    clustering->train(n, x, index);
    return 0;
  }

  void faiss_Clustering_get_centroids(faiss::Clustering* clustering, float* out) {
    size_t size = clustering->k * clustering->d;
    for (size_t i = 0; i < size; i++) {
      out[i] = clustering->centroids[i];
    }
  }

  void faiss_Clustering_assign(faiss::Clustering* clustering, int n, const float* x, int64_t* labels) {
    faiss::IndexFlatL2 index(clustering->d);
    index.add(clustering->k, clustering->centroids.data());

    std::vector<float> distances(n);
    index.search(n, x, 1, distances.data(), labels);
  }

  void faiss_Clustering_free(faiss::Clustering* clustering) {
    delete clustering;
  }
}
