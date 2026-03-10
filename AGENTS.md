# Agent Guidelines

## Before Committing

1. **Run linting**: `deno lint` must pass with no errors
2. **Run tests**: `deno task test` must pass

## Code Style

- Use bare specifiers for imports (add dependencies to `deno.json` imports)
- Avoid inline `jsr:`, `npm:`, or `https:` imports in source files
