/**
 * Semantic search demo using Ollama embeddings.
 *
 * Requires a running Ollama server: https://ollama.com/
 * Default model: nomic-embed-text (run `ollama pull nomic-embed-text` first)
 *
 * Usage:
 *   deno run --allow-ffi --allow-net=localhost examples/search.ts
 */
import { FaissIndex } from "../mod.ts";

const OLLAMA_URL = "http://localhost:11434/api/embeddings";
const MODEL = "nomic-embed-text";

async function embed(text: string): Promise<Float32Array<ArrayBuffer>> {
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

const sentences = [
  "Cats are small, carnivorous mammals.",
  "Domestic cats are often valued by humans for companionship.",
  "Cats are known for their agility and playful behavior.",
  "A group of cats is called a clowder.",
  "Cats have a unique way of communicating with humans.",
  "The average lifespan of a domestic cat is around 15 years.",
  "Cats have retractable claws that help them catch prey.",
  "There are many different breeds of cats with various characteristics.",
  "Cats are known for their grooming habits.",
  "Many cultures consider cats to be symbols of mystery and independence.",
  "Cats are known to be territorial animals.",
  "Cats have a strong sense of smell and hearing.",
  "Cats are known to be excellent hunters.",
  "Cats are often depicted in art and literature.",
  "Cats are popular pets around the world.",
  "Cats are known to be curious animals.",
  "Cats are often associated with superstitions.",
  "Cats are known to be nocturnal animals.",
  "Cats are known to be independent animals.",
];

const queryText = "Cats have claws that help them hunt.";

console.log(`Embedding ${sentences.length} sentences...`);
const embeddings = await Promise.all(sentences.map(embed));

const index = new FaissIndex(embeddings[0].length);
for (const emb of embeddings) {
  index.addVectors(emb);
}

const queryVector = await embed(queryText);
const k = Math.ceil(sentences.length / 2);
const { distances, indices } = index.search(queryVector, k);

console.log(`\nQuery: "${queryText}"\n`);
console.log("Top results:");
for (let i = 0; i < indices.length; i++) {
  console.log(`  [${i + 1}] "${sentences[indices[i]]}" (distance: ${distances[i].toFixed(4)})`);
}

index.close();
