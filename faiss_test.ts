import { assertEquals, assertThrows } from "jsr:@std/assert";
import { FaissIndex } from "./mod.ts";

Deno.test("search returns correct number of results", () => {
  const index = new FaissIndex(3);
  index.addVectors(new Float32Array([1, 0, 0]));
  index.addVectors(new Float32Array([0, 1, 0]));
  index.addVectors(new Float32Array([0, 0, 1]));

  const { distances, indices } = index.search(new Float32Array([1, 0, 0]), 2);

  assertEquals(distances.length, 2);
  assertEquals(indices.length, 2);

  index.close();
});

Deno.test("exact match returns distance 0", () => {
  const index = new FaissIndex(3);
  index.addVectors(new Float32Array([1, 0, 0]));
  index.addVectors(new Float32Array([0, 1, 0]));

  const { distances, indices } = index.search(new Float32Array([1, 0, 0]), 1);

  assertEquals(distances[0], 0);
  assertEquals(indices[0], 0);

  index.close();
});

Deno.test("k larger than index size returns all vectors", () => {
  const index = new FaissIndex(2);
  index.addVectors(new Float32Array([1, 0]));
  index.addVectors(new Float32Array([0, 1]));

  const { distances, indices } = index.search(new Float32Array([1, 0]), 100);

  assertEquals(distances.length, 2);
  assertEquals(indices.length, 2);

  index.close();
});

Deno.test("search on empty index returns empty results", () => {
  const index = new FaissIndex(3);

  const { distances, indices } = index.search(new Float32Array([1, 0, 0]), 3);

  assertEquals(distances.length, 0);
  assertEquals(indices.length, 0);

  index.close();
});

Deno.test("addVectors throws on wrong dimensions", () => {
  const index = new FaissIndex(3);

  assertThrows(
    () => index.addVectors(new Float32Array([1, 0])), // 2 values, not divisible by 3
    Error,
  );

  index.close();
});

Deno.test("search throws on wrong query dimensions", () => {
  const index = new FaissIndex(3);
  index.addVectors(new Float32Array([1, 0, 0]));

  assertThrows(
    () => index.search(new Float32Array([1, 0]), 1), // wrong dims
    Error,
    "3 dimensions",
  );

  index.close();
});

Deno.test("close cleans up without error", () => {
  const index = new FaissIndex(2);
  index.addVectors(new Float32Array([1, 0]));
  index.close();
  // double-close should not throw
  index.close();
});
