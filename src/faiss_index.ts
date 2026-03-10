import { dylibPath } from "./libfaiss_wrapper.ts";

const { UnsafePointer } = Deno;

interface faiss_IndexFlatL2_search {
  (
    index: Deno.UnsafePointer,
    n: bigint,
    x: Deno.UnsafePointer,
    k: bigint,
    distances: Deno.UnsafePointer,
    indices: Deno.UnsafePointer,
  ): void;
}

interface faiss_IndexFlatL2_add {
  (index: Deno.UnsafePointer, n: bigint, x: Deno.UnsafePointer): void;
}

interface faiss_IndexFlatL2_new {
  (d: number): Deno.UnsafePointer;
}

interface faiss_IndexFlatL2_free {
  (index: Deno.UnsafePointer): void;
}

function bigInt64ArrayToInt32Array(bigIntArray: BigInt64Array): Int32Array {
  const int32Array = new Int32Array(bigIntArray.length);
  for (let i = 0; i < bigIntArray.length; i++) {
    const val = bigIntArray[i];
    if (
      val > BigInt(Number.MAX_SAFE_INTEGER) ||
      val < BigInt(Number.MIN_SAFE_INTEGER)
    ) {
      throw new Error(`Index value out of safe integer range: ${val}`);
    }
    int32Array[i] = Number(val);
  }
  return int32Array;
}

interface FaissLib {
  close: () => void;
  symbols: {
    faiss_IndexFlatL2_search: faiss_IndexFlatL2_search;
    faiss_IndexFlatL2_add: faiss_IndexFlatL2_add;
    faiss_IndexFlatL2_new: faiss_IndexFlatL2_new;
    faiss_IndexFlatL2_free: faiss_IndexFlatL2_free;
  };
}

/**
 * A Deno FFI wrapper around FAISS `IndexFlatL2`.
 *
 * Performs exact (brute-force) nearest-neighbor search using squared L2 distance.
 * All vectors must have the same dimensionality specified at construction time.
 */
class FaissIndex {
  private lib: FaissLib | null;
  private indexPtr: Deno.UnsafePointer | null = null;

  // number of vectors currently in the index
  private numVectors: number;

  // dimensionality of each vector
  private dimensions: number;

  /**
   * Creates a new FAISS IndexFlatL2 for vectors of the given dimensionality.
   * @param dimensionsIn - Number of float32 components per vector.
   */
  constructor(private dimensionsIn: number) {
    const symbols: Deno.ForeignLibraryInterface = {
      faiss_IndexFlatL2_new: {
        parameters: ["i32"],
        result: "pointer",
      },
      faiss_IndexFlatL2_add: {
        parameters: ["pointer", "i64", "pointer"],
        result: "void",
      },
      faiss_IndexFlatL2_search: {
        parameters: ["pointer", "i64", "pointer", "i64", "pointer", "pointer"],
        result: "void",
      },
      faiss_IndexFlatL2_free: {
        parameters: ["pointer"],
        result: "void",
      },
    };
    this.numVectors = 0;
    this.dimensions = dimensionsIn;

    this.lib = Deno.dlopen(
      dylibPath,
      symbols,
    ) as unknown as FaissLib;
    this.indexPtr = this.lib.symbols.faiss_IndexFlatL2_new(this.dimensions);
  }

  /**
   * Adds one or more vectors to the index.
   * @param vectors - Flat Float32Array; length must be a multiple of `dimensions`.
   */
  public addVectors(vectors: Float32Array<ArrayBuffer>) {
    if (!this.indexPtr || !this.lib) {
      throw new Error("Index not initialized");
    }

    const vectorPtr = UnsafePointer.of(vectors);
    if (!vectorPtr) {
      throw new Error("Failed to create vector pointer");
    }

    const numVectors = vectors.length / this.dimensions;
    if (numVectors <= 0) {
      throw new Error("The number of vectors should be at least one");
    }

    if (!Number.isInteger(numVectors)) {
      throw new Error(
        `Vector length (${vectors.length}) not divisible by dimensions (${this.dimensions})`,
      );
    }

    this.lib.symbols.faiss_IndexFlatL2_add(
      this.indexPtr,
      BigInt(numVectors),
      vectorPtr,
    );

    this.numVectors += numVectors;

    return true;
  }

  /**
   * Finds the k nearest neighbors of `query`.
   * Returns fewer than k results if the index contains fewer vectors.
   * @param query - Query vector; must have exactly `dimensions` components.
   * @param k - Maximum number of results to return.
   */
  public search(query: Float32Array<ArrayBuffer>, k: number) {
    if (!this.indexPtr || !this.lib) {
      throw new Error("Index not initialized");
    }

    if (query.length !== this.dimensions) {
      throw new Error(`Query vector must have ${this.dimensions} dimensions`);
    }

    const numResults = Math.min(this.numVectors, k);

    if (numResults <= 0) {
      return { distances: new Float32Array(0), indices: new Int32Array(0) };
    }

    const distances = new Float32Array(numResults);
    const indices = new BigInt64Array(numResults);

    const queryPtr = UnsafePointer.of(query);
    const distancesPtr = UnsafePointer.of(distances);
    const indicesPtr = UnsafePointer.of(indices);

    if (!queryPtr || !distancesPtr || !indicesPtr) {
      throw new Error("Failed to create pointers for search operation");
    }

    this.lib.symbols.faiss_IndexFlatL2_search(
      this.indexPtr,
      BigInt(1),
      queryPtr,
      BigInt(numResults),
      distancesPtr,
      indicesPtr,
    );

    const safeIndices = bigInt64ArrayToInt32Array(indices);

    return { distances, indices: safeIndices };
  }

  /**
   * Frees the native FAISS index and closes the shared library.
   * Safe to call multiple times.
   */
  public close() {
    if (this.indexPtr && this.lib) {
      this.lib.symbols.faiss_IndexFlatL2_free(this.indexPtr);
      this.indexPtr = null;
    }

    if (this.lib) {
      this.lib.close();
      this.lib = null;
    }
  }
}

interface ClusteringLib {
  close: () => void;
  symbols: {
    faiss_Clustering_new: (d: number, k: number, niter: number) => Deno.PointerValue;
    faiss_Clustering_train: (clustering: Deno.PointerValue, n: number, x: Deno.PointerValue) => number;
    faiss_Clustering_get_centroids: (clustering: Deno.PointerValue, out: Deno.PointerValue) => void;
    faiss_Clustering_assign: (clustering: Deno.PointerValue, n: number, x: Deno.PointerValue, labels: Deno.PointerValue) => void;
    faiss_Clustering_free: (clustering: Deno.PointerValue) => void;
  };
}

interface FaissClusteringOptions {
  niter?: number;
}

/**
 * A Deno FFI wrapper around FAISS `Clustering`.
 *
 * Performs k-means clustering to group vectors into k clusters and compute centroids.
 * All vectors must have the same dimensionality specified at construction time.
 */
class FaissClustering {
  private lib: ClusteringLib | null;
  private clusteringPtr: Deno.PointerValue | null = null;
  private dimensions: number;
  private numClusters: number;
  private trained: boolean = false;

  /**
   * Creates a new FAISS Clustering object.
   * @param dimensions - Number of float32 components per vector.
   * @param numClusters - Number of clusters (k) to create.
   * @param options - Optional configuration (niter: number of k-means iterations, default 25).
   */
  constructor(dimensions: number, numClusters: number, options?: FaissClusteringOptions) {
    const niter = options?.niter ?? 25;

    const symbols: Deno.ForeignLibraryInterface = {
      faiss_Clustering_new: {
        parameters: ["i32", "i32", "i32"],
        result: "pointer",
      },
      faiss_Clustering_train: {
        parameters: ["pointer", "i32", "pointer"],
        result: "i32",
      },
      faiss_Clustering_get_centroids: {
        parameters: ["pointer", "pointer"],
        result: "void",
      },
      faiss_Clustering_assign: {
        parameters: ["pointer", "i32", "pointer", "pointer"],
        result: "void",
      },
      faiss_Clustering_free: {
        parameters: ["pointer"],
        result: "void",
      },
    };

    this.dimensions = dimensions;
    this.numClusters = numClusters;

    this.lib = Deno.dlopen(dylibPath, symbols) as unknown as ClusteringLib;
    this.clusteringPtr = this.lib.symbols.faiss_Clustering_new(dimensions, numClusters, niter);
  }

  /**
   * Trains the clustering on the provided vectors.
   * @param vectors - Flat Float32Array; length must be a multiple of `dimensions`.
   */
  public train(vectors: Float32Array<ArrayBuffer>): void {
    if (!this.clusteringPtr || !this.lib) {
      throw new Error("Clustering not initialized");
    }

    const numVectors = vectors.length / this.dimensions;
    if (!Number.isInteger(numVectors)) {
      throw new Error(
        `Vector length (${vectors.length}) not divisible by dimensions (${this.dimensions})`,
      );
    }

    if (numVectors < this.numClusters) {
      throw new Error(
        `Number of training vectors (${numVectors}) must be at least the number of clusters (${this.numClusters})`,
      );
    }

    const vectorPtr = UnsafePointer.of(vectors);
    if (!vectorPtr) {
      throw new Error("Failed to create vector pointer");
    }

    this.lib.symbols.faiss_Clustering_train(this.clusteringPtr, numVectors, vectorPtr);
    this.trained = true;
  }

  /**
   * Returns the computed centroids after training.
   * @returns Array of centroid vectors, one Float32Array per cluster.
   */
  public getCentroids(): Float32Array[] {
    if (!this.clusteringPtr || !this.lib) {
      throw new Error("Clustering not initialized");
    }

    if (!this.trained) {
      throw new Error("Clustering must be trained before getting centroids");
    }

    const flat = new Float32Array(this.numClusters * this.dimensions);
    const flatPtr = UnsafePointer.of(flat);
    if (!flatPtr) {
      throw new Error("Failed to create centroids pointer");
    }

    this.lib.symbols.faiss_Clustering_get_centroids(this.clusteringPtr, flatPtr);

    const centroids: Float32Array[] = [];
    for (let i = 0; i < this.numClusters; i++) {
      centroids.push(flat.slice(i * this.dimensions, (i + 1) * this.dimensions));
    }
    return centroids;
  }

  /**
   * Assigns each input vector to its nearest cluster.
   * @param vectors - Flat Float32Array; length must be a multiple of `dimensions`.
   * @returns Int32Array of cluster indices, one per input vector.
   */
  public assign(vectors: Float32Array<ArrayBuffer>): Int32Array {
    if (!this.clusteringPtr || !this.lib) {
      throw new Error("Clustering not initialized");
    }

    if (!this.trained) {
      throw new Error("Clustering must be trained before assigning vectors");
    }

    const numVectors = vectors.length / this.dimensions;
    if (!Number.isInteger(numVectors)) {
      throw new Error(
        `Vector length (${vectors.length}) not divisible by dimensions (${this.dimensions})`,
      );
    }

    const vectorPtr = UnsafePointer.of(vectors);
    const labels = new BigInt64Array(numVectors);
    const labelsPtr = UnsafePointer.of(labels);

    if (!vectorPtr || !labelsPtr) {
      throw new Error("Failed to create pointers for assign operation");
    }

    this.lib.symbols.faiss_Clustering_assign(this.clusteringPtr, numVectors, vectorPtr, labelsPtr);

    return bigInt64ArrayToInt32Array(labels);
  }

  /**
   * Frees the native FAISS clustering object and closes the shared library.
   * Safe to call multiple times.
   */
  public close(): void {
    if (this.clusteringPtr && this.lib) {
      this.lib.symbols.faiss_Clustering_free(this.clusteringPtr);
      this.clusteringPtr = null;
    }

    if (this.lib) {
      this.lib.close();
      this.lib = null;
    }
  }
}

/**
 * Computes the squared L2 (Euclidean) distance between two vectors.
 * This matches the distance metric used by FaissIndex and FaissClustering.
 * @param a - First vector
 * @param b - Second vector (must have same length as a)
 * @returns Squared L2 distance
 */
function l2Distance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Vector lengths must match: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

export { FaissIndex, FaissClustering, l2Distance };
