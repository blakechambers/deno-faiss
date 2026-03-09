# deno-faiss

Deno FFI bindings for [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search). Wraps `IndexFlatL2` for exact nearest-neighbor search over float32 vectors.

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

Run with the required permissions:

```sh
deno run --allow-ffi examples/simple.ts
```

## Examples

- [`examples/simple.ts`](examples/simple.ts) — basic add/search demo
- [`examples/embeds.ts`](examples/embeds.ts) — semantic search using Ollama embeddings (requires a running [Ollama](https://ollama.com/) server)

Run the Ollama example:

```sh
deno run --allow-ffi --allow-net=localhost examples/embeds.ts
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

## Permissions

Requires `--allow-ffi` (FFI was stabilized in Deno 2.0; `--unstable-ffi` is no longer needed).
