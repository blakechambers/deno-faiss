/**
 * Semantic clustering demo using Ollama embeddings.
 *
 * Groups sentences into clusters based on their meaning using k-means.
 *
 * Requires a running Ollama server: https://ollama.com/
 * Default model: nomic-embed-text (run `ollama pull nomic-embed-text` first)
 *
 * Usage:
 *   deno run --allow-ffi --allow-net=localhost examples/clustering.ts
 */
import { FaissClustering } from "../mod.ts";

const OLLAMA_URL = "http://localhost:11434/api/embeddings";
// const MODEL = "nomic-embed-text";
const MODEL = "qwen3-embedding:4b";

async function embed(text: string): Promise<Float32Array> {
  const res = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, prompt: text }),
  });

  if (!res.ok) {
    throw new Error(`Ollama request failed: ${res.status} ${res.statusText}`);
  }

  const { embedding } = await res.json();
  return new Float32Array(embedding);
}

// Sentences about different topics that should naturally cluster
const sentences = [
  // Animals
  "Cats are small, carnivorous mammals.",
  "Dogs are loyal companions to humans.",
  "Elephants are the largest land animals.",
  "Birds can fly using their wings.",
  "Fish breathe through their gills underwater.",
  "Lions are known as the king of the jungle.",

  // Food & Cooking
  "Pizza is a popular Italian dish with cheese and tomato sauce.",
  "Sushi is a traditional Japanese food made with rice and fish.",
  "Chocolate is made from roasted cacao beans.",
  "Fresh vegetables are essential for a healthy diet.",
  "Bread is baked from flour, water, and yeast.",
  "Coffee is one of the most popular beverages worldwide.",

  // Technology
  "Computers process data using electronic circuits.",
  "Smartphones have revolutionized communication.",
  "Artificial intelligence is transforming many industries.",
  "The internet connects billions of devices globally.",
  "Programming languages are used to write software.",
  "Cloud computing enables remote data storage.",

  // Sports
  "Soccer is the most popular sport in the world.",
  "Basketball was invented by James Naismith in 1891.",
  "Tennis is played on grass, clay, or hard courts.",
  "Swimming is both a sport and a survival skill.",
  "Golf requires precision and patience.",
  "Running marathons tests human endurance.",
];

const NUM_CLUSTERS = 10;

console.log(`Embedding ${sentences.length} sentences...`);
const embeddings = await Promise.all(sentences.map(embed));
const dimensions = embeddings[0].length;

// Concatenate all embeddings into a single Float32Array
const allVectors = new Float32Array(sentences.length * dimensions);
for (let i = 0; i < embeddings.length; i++) {
  allVectors.set(embeddings[i], i * dimensions);
}

console.log(`Training k-means with ${NUM_CLUSTERS} clusters...\n`);
const clustering = new FaissClustering(dimensions, NUM_CLUSTERS, { niter: 25 });
clustering.train(allVectors);

// Assign sentences to clusters
const assignments = clustering.assign(allVectors);

console.log("Cluster assignments:", clustering.getCentroids());

// Group sentences by cluster
const clusters: string[][] = Array.from({ length: NUM_CLUSTERS }, () => []);
for (let i = 0; i < sentences.length; i++) {
  clusters[assignments[i]].push(sentences[i]);
}

// Display results
console.log("Clustering Results");
console.log("==================\n");

for (let i = 0; i < NUM_CLUSTERS; i++) {
  console.log(`Cluster ${i + 1} (${clusters[i].length} sentences):`);
  for (const sentence of clusters[i]) {
    console.log(`  - ${sentence}`);
  }
  console.log();
}

clustering.close();
