import { assertEquals, assertThrows } from "@std/assert";
import { FaissIndex, FaissClustering, l2Distance } from "./mod.ts";

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

// FaissClustering tests

Deno.test("clustering returns correct centroid dimensions", () => {
  const dimensions = 3;
  const numClusters = 2;
  const clustering = new FaissClustering(dimensions, numClusters);

  // Create 4 vectors in 2 clear groups
  const vectors = new Float32Array([
    1, 0, 0,  // group 1
    1.1, 0.1, 0,  // group 1
    0, 1, 0,  // group 2
    0.1, 1.1, 0,  // group 2
  ]);

  clustering.train(vectors);
  const centroids = clustering.getCentroids();

  assertEquals(centroids.length, numClusters);
  assertEquals(centroids[0].length, dimensions);
  assertEquals(centroids[1].length, dimensions);
  clustering.close();
});

Deno.test("clustering assigns vectors to clusters", () => {
  const dimensions = 2;
  const numClusters = 2;
  const clustering = new FaissClustering(dimensions, numClusters);

  // Two distinct groups
  const vectors = new Float32Array([
    0, 0,
    0.1, 0.1,
    10, 10,
    10.1, 10.1,
  ]);

  clustering.train(vectors);
  const assignments = clustering.assign(vectors);

  assertEquals(assignments.length, 4);
  // Vectors 0 and 1 should be in the same cluster
  assertEquals(assignments[0], assignments[1]);
  // Vectors 2 and 3 should be in the same cluster
  assertEquals(assignments[2], assignments[3]);
  // The two groups should be in different clusters
  if (assignments[0] === assignments[2]) {
    throw new Error("Different groups should be in different clusters");
  }

  clustering.close();
});

Deno.test("clustering throws on wrong dimensions", () => {
  const clustering = new FaissClustering(3, 2);

  assertThrows(
    () => clustering.train(new Float32Array([1, 0])), // 2 values, not divisible by 3
    Error,
  );

  clustering.close();
});

Deno.test("clustering throws when not enough vectors for clusters", () => {
  const clustering = new FaissClustering(2, 5); // 5 clusters

  assertThrows(
    () => clustering.train(new Float32Array([1, 0, 0, 1])), // only 2 vectors
    Error,
    "at least the number of clusters",
  );

  clustering.close();
});

Deno.test("clustering throws when getting centroids before training", () => {
  const clustering = new FaissClustering(2, 2);

  assertThrows(
    () => clustering.getCentroids(),
    Error,
    "must be trained",
  );

  clustering.close();
});

Deno.test("clustering throws when assigning before training", () => {
  const clustering = new FaissClustering(2, 2);

  assertThrows(
    () => clustering.assign(new Float32Array([1, 0])),
    Error,
    "must be trained",
  );

  clustering.close();
});

Deno.test("clustering close cleans up without error", () => {
  const clustering = new FaissClustering(2, 2);
  clustering.close();
  // double-close should not throw
  clustering.close();
});

Deno.test("clustering with custom niter", () => {
  const clustering = new FaissClustering(2, 2, { niter: 10 });

  const vectors = new Float32Array([
    0, 0,
    0.1, 0.1,
    10, 10,
    10.1, 10.1,
  ]);

  clustering.train(vectors);
  const centroids = clustering.getCentroids();

  assertEquals(centroids.length, 2); // 2 clusters
  assertEquals(centroids[0].length, 2); // 2 dimensions each
  clustering.close();
});

// l2Distance tests

Deno.test("l2Distance returns 0 for identical vectors", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([1, 2, 3]);
  assertEquals(l2Distance(a, b), 0);
});

Deno.test("l2Distance computes correct squared distance", () => {
  const a = new Float32Array([0, 0, 0]);
  const b = new Float32Array([3, 4, 0]);
  // 3^2 + 4^2 + 0^2 = 9 + 16 = 25
  assertEquals(l2Distance(a, b), 25);
});

Deno.test("l2Distance matches FAISS search distance", () => {
  const index = new FaissIndex(3);
  const vector = new Float32Array([1, 2, 3]);
  index.addVectors(vector);

  const query = new Float32Array([4, 6, 3]);
  const { distances } = index.search(query, 1);

  const manualDistance = l2Distance(vector, query);
  assertEquals(distances[0], manualDistance);

  index.close();
});

Deno.test("l2Distance throws on mismatched lengths", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([1, 2]);

  assertThrows(
    () => l2Distance(a, b),
    Error,
    "must match",
  );
});
