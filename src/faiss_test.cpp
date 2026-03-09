#include <iostream>
#include <faiss/IndexFlat.h>

// Declare the functions from the wrapper
extern "C" {
  faiss::IndexFlatL2* faiss_IndexFlatL2_new(int d);
  void faiss_IndexFlatL2_add(faiss::IndexFlatL2* index, int64_t n, const float* x);
  void faiss_IndexFlatL2_search(faiss::IndexFlatL2* index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
  void faiss_IndexFlatL2_free(faiss::IndexFlatL2* index);
}

// Test function
void test_faiss() {
    int d = 4; // Dimension of vectors
    int n = 5; // Number of vectors to add

    // Create the index
    faiss::IndexFlatL2* index = faiss_IndexFlatL2_new(d);
    if (!index) {
        std::cerr << "Failed to create FAISS index." << std::endl;
        return;
    }

    // Create some example vectors (5 vectors, 4 dimensions each)
    float xb[5 * 4] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };

    // Add vectors to the index
    faiss_IndexFlatL2_add(index, n, xb);

    // Create a query vector
    float xq[4] = { 0.0, 1.0, 0.0, 0.0 };

    // Prepare results storage
    int k = 3; // Number of nearest neighbors to search for
    float distances[3];
    int64_t labels[3];

    // Perform the search
    faiss_IndexFlatL2_search(index, 1, xq, k, distances, labels);

    // Print the results
    std::cout << "Distances: ";
    for (int i = 0; i < k; ++i) {
        std::cout << distances[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Labels: ";
    for (int i = 0; i < k; ++i) {
        std::cout << labels[i] << " ";
    }
    std::cout << std::endl;

    // Free the index
    faiss_IndexFlatL2_free(index);
}

int main() {
    test_faiss();
    return 0;
}
