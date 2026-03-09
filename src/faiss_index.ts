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
  private lib: FaissLib;
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
    if (!this.indexPtr) {
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
    if (!this.indexPtr) {
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
    if (this.indexPtr) {
      this.lib.symbols.faiss_IndexFlatL2_free(this.indexPtr);
      this.indexPtr = null;
    }

    if (this.lib) {
      this.lib.close();
    }
  }
}

export { FaissIndex };
