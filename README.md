# nervx

A codebase brain for AI coding assistants. Pre-indexed navigation, blast radius analysis, dead code detection, and architectural pattern recognition — all from a single `pip install`.

## Install

```bash
pip install nervx
```

## Quick Start

```bash
# Build the brain for your project
nervx build .

# Ask questions in natural language
nervx nav "how does authentication work"

# Check blast radius before refactoring
nervx blast-radius "src/auth.py::validate_token"

# Find dead code
nervx find --dead

# Find untested critical code
nervx find --no-tests --importance-gt 20

# Open interactive visualization
nervx viz .
```

## What It Does

nervx parses your codebase with tree-sitter, builds a graph of every function, class, and method, then pre-computes:

- **Edges**: who calls what, who imports what, who inherits from what
- **Importance scores**: based on caller count, cross-module usage, and connectivity
- **Architectural patterns**: factories, singletons, event buses, strategy patterns, repositories
- **Concept paths**: end-to-end call chains and domain clusters
- **Git intelligence**: hotspots, temporal coupling, churn analysis
- **Contract analysis**: callers that disagree on error handling
- **Dead code**: unreferenced functions and classes

All stored in a single SQLite database (`.nervx/brain.db`), queryable in milliseconds.

## Commands

| Command | What it does |
|---------|-------------|
| `nervx build <path>` | Full build of the brain |
| `nervx update <path>` | Incremental update (only changed files) |
| `nervx nav "<question>"` | Natural language navigation with execution flows |
| `nervx blast-radius "<symbol>"` | Impact analysis for refactoring |
| `nervx find --dead` | Find unreferenced symbols |
| `nervx find --no-tests` | Find untested code |
| `nervx flows [keyword]` | Show execution paths |
| `nervx diff --days 7` | Recent structural changes |
| `nervx viz .` | Interactive D3 visualization |
| `nervx stats` | Graph statistics |

## Claude Code Integration

When you run `nervx build`, it automatically adds instructions to your project's `CLAUDE.md` that teach Claude Code to use nervx commands. Claude will use `nervx nav` before exploring code, check blast radius before refactoring, and find dead code before cleanup — saving tokens and tool calls.

## Supported Languages

Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, Ruby

## Watch Mode (Optional)

```bash
pip install nervx[watch]
nervx watch .
```

Auto-updates the brain when files change.

## License

MIT
