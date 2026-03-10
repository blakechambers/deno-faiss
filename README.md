# deno-faiss

Deno FFI bindings for [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search). Wraps `IndexFlatL2` for exact nearest-neighbor search and `Clustering` for k-means clustering over float32 vectors.

## Prerequisites

- [Deno](https://deno.land/) 1.x or later
- FAISS installed:
  - **macOS**: `brew install faiss`
  - **Linux**: install via your distro's package manager or build from source

## Build

Compile the thin C++ wrapper against your locally installed FAISS:

```sh
./build.sh
```

This places the shared library under `native/<os>-<arch>/`.

A pre-built `native/darwin-aarch64/libfaiss_wrapper.dylib` is included for Apple Silicon users who don't want to build from source.

## Usage

### Nearest-Neighbor Search

```ts
import { FaissIndex } from "./mod.ts";

const index = new FaissIndex(3); // 3-dimensional vectors

index.addVectors(new Float32Array([1, 0, 0]));
index.addVectors(new Float32Array([0, 1, 0]));
index.addVectors(new Float32Array([0, 0, 1]));

const { distances, indices } = index.search(
  new Float32Array([1, 0, 0]), // query vector
  2,                           // k: number of results
);

console.log(distances); // Float32Array of L2 distances
console.log(indices);   // Int32Array of vector indices

index.close();
```

### K-Means Clustering

```ts
import { FaissClustering } from "./mod.ts";

const clustering = new FaissClustering(3, 2); // 3-dim vectors, 2 clusters

// Train on sample vectors
const vectors = new Float32Array([
  1, 0, 0,
  0.9, 0.1, 0,
  0, 1, 0,
  0.1, 0.9, 0,
]);
clustering.train(vectors);

// Assign vectors to clusters
const assignments = clustering.assign(vectors);
console.log(assignments); // Int32Array of cluster indices

// Get cluster centroids
const centroids = clustering.getCentroids();
console.log(centroids); // Array of Float32Array centroids

clustering.close();
```

Run with the required permissions:

```sh
deno run --allow-ffi examples/simple.ts
```

## Examples

- [`examples/simple.ts`](examples/simple.ts) — basic add/search demo
- [`examples/search.ts`](examples/search.ts) — semantic search using Ollama embeddings
- [`examples/clustering.ts`](examples/clustering.ts) — k-means clustering with Ollama embeddings

The Ollama examples require a running [Ollama](https://ollama.com/) server:

```sh
# Semantic search
deno run --allow-ffi --allow-net=localhost examples/search.ts

# Clustering
deno run --allow-ffi --allow-net=localhost examples/clustering.ts
```

## API

### `new FaissIndex(dimensions: number)`

Creates a new `IndexFlatL2` index for vectors of the given dimensionality.

### `addVectors(vectors: Float32Array): boolean`

Adds one or more vectors to the index. The array length must be a multiple of `dimensions`.

### `search(query: Float32Array, k: number): { distances: Float32Array, indices: Int32Array }`

Returns the `k` nearest neighbors to `query`. Returns fewer results if the index contains fewer than `k` vectors.

### `close(): void`

Frees the native index and closes the shared library. Call this when done.

---

### `new FaissClustering(dimensions: number, numClusters: number, options?: { niter?: number })`

Creates a new k-means clustering object.

- `dimensions` — Number of float32 components per vector
- `numClusters` — Number of clusters (k) to create
- `options.niter` — Number of k-means iterations (default: 25)

### `train(vectors: Float32Array): void`

Trains the clustering on the provided vectors. The number of vectors must be at least `numClusters`.

### `assign(vectors: Float32Array): Int32Array`

Assigns each vector to its nearest cluster. Returns an array of cluster indices.

### `getCentroids(): Float32Array[]`

Returns the computed centroid vectors after training.

### `close(): void`

Frees the native clustering object and closes the shared library.

---

### `l2Distance(a: Float32Array, b: Float32Array): number`

Utility function that computes the squared L2 (Euclidean) distance between two vectors. This matches the distance metric used by `FaissIndex` and `FaissClustering`.

## Permissions

Requires `--allow-ffi` (FFI was stabilized in Deno 2.0; `--unstable-ffi` is no longer needed).
