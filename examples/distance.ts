/**
 * Demonstrates that `l2Distance` matches the distance metric used by `FaissIndex.search`.
 *
 * Usage:
 *   deno run --allow-ffi examples/distance.ts
 */
import { FaissIndex, l2Distance } from "../mod.ts";

const vectors = [
  new Float32Array([1.0, 0.0, 0.0, 0.0]),
  new Float32Array([0.0, 1.0, 0.0, 0.0]),
  new Float32Array([0.5, 0.5, 0.0, 0.0]),
  new Float32Array([0.9, 0.1, 0.0, 0.0]),
];

const index = new FaissIndex(4);
for (const v of vectors) {
  index.addVectors(v);
}

const query = new Float32Array([1.0, 0.0, 0.0, 0.0]);
const { distances, indices } = index.search(query, vectors.length);

console.log("rank  idx  search distance   l2Distance(query, vec)");
for (let i = 0; i < indices.length; i++) {
  const manual = l2Distance(query, vectors[indices[i]]);
  console.log(
    `  ${i + 1}    ${indices[i]}    ${distances[i].toFixed(6)}         ${manual.toFixed(6)}`,
  );
}

index.close();
