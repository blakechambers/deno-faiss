import { FaissIndex } from "../mod.ts";

const index = new FaissIndex(12);

const vector = new Float32Array([
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
  0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
]);

index.addVectors(vector);
index.addVectors(vector);
index.addVectors(vector);

const query = new Float32Array([
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
  0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
]);

const { distances, indices } = index.search(query, 3);

console.log("distances:", distances);
console.log("indices:", indices);

index.close();
