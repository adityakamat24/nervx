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

# 50-token preview of a symbol — signature, callees, caller count, no source
nervx peek validate_token

# Structural overview of a file — ~150 tokens instead of 4000
nervx tree src/auth.py

# Read a function's source + everything it calls
nervx read validate_token --context 1

# Quick yes/no: does A call B?
nervx ask calls login validate_token

# Confirm a call path with BFS (up to 6 hops)
nervx verify "login calls validate_token"

# Shortest call path between two symbols
nervx trace login validate_token

# Find where a string literal appears across all languages
nervx string-refs "user_id"

# Check blast radius before refactoring
nervx blast-radius validate_token

# Find dead code (framework-aware — skips decorated handlers)
nervx find --dead

# Run pytest with compact output (~80 tokens vs 8000 of traceback)
nervx run pytest tests/

# Open interactive visualization
nervx viz .
```

## Benchmarks

Tested on [FastAPI](https://github.com/fastapi/fastapi) — 3 identical questions asked with and without nervx:

| Metric | Without nervx | With nervx | Reduction |
|--------|--------------|------------|-----------|
| Tool calls | 93 | 56 | **-40%** |
| Output tokens | 15,694 | 8,196 | **-48%** |
| Grep searches | 63 | 22 | **-65%** |
| API calls | 115 | 73 | **-37%** |
| Peak context | 70,503 | 57,141 | **-19%** |

nervx replaces dozens of blind grep/read cycles with pre-indexed lookups. Fewer tool calls, less token waste, faster answers.

## What It Does

nervx parses your codebase with tree-sitter, builds a graph of every function, class, and method, then pre-computes:

- **Edges**: who calls what, who imports what, who inherits from what, and — new in 0.2.2 — which base-class methods dispatch to which concrete overrides (`dispatches_to`), so `trace` can follow polymorphic calls that static resolution misses
- **Importance scores**: weighted per edge type (`calls`×2, `inherits`×1.5, `imports`×0.5) with a 0–100 percentile rank
- **Path categories**: every node tagged `category:{vendor,generated,test,example,doc,script,core}` so `find`/`nav`/`blast-radius` can drop noise with `--exclude-category` — new in 0.2.4, `script` catches `scripts/`, `ci/`, `build_scripts/`, `.github/` paths that otherwise pollute importance rankings
- **Architectural patterns**: factories, singletons, event buses, strategy patterns, repositories
- **Concept paths**: end-to-end call chains and domain clusters
- **Git intelligence**: hotspots, temporal coupling, churn analysis — `cochange --why` exposes the actual commit hashes behind each coupling
- **Contract analysis**: callers that disagree on error handling
- **Warning provenance**: every warning carries its methodology, confidence, and evidence (visible via `nav --verbose-warnings` or `--json`)
- **Dead code**: unreferenced functions and classes

All stored in a single SQLite database (`.nervx/brain.db`), queryable in milliseconds.

## Commands

### Build
| Command | What it does |
|---------|-------------|
| `nervx build <path>` | Full build of the brain |
| `nervx update <path>` | Incremental update (only changed files) |
| `nervx watch <path>` | Auto-update on file changes (requires `nervx[watch]`) |

### Exploration
| Command | What it does |
|---------|-------------|
| `nervx nav "<question>"` | Natural language navigation with execution flows |
| `nervx tree <file>` | Structural overview of a file (~150 tokens) |
| `nervx peek <symbol>` | 50-token preview — signature, callees, caller count |

### Reading
| Command | What it does |
|---------|-------------|
| `nervx read <symbol>` | Source of one function/method |
| `nervx read <symbol> --context 1` | Symbol source + everything it calls |
| `nervx read <symbol> --since <hash>` | Returns "unchanged" if symbol hasn't been edited |

### Quick Answers (5–30 tokens each)
| Command | Answers |
|---------|---------|
| `nervx ask exists <symbol>` | yes / no |
| `nervx ask signature <symbol>` | the function signature |
| `nervx ask calls <A> <B>` | does A call B directly? |
| `nervx ask imports <file>` | what this file imports |
| `nervx ask is-async <symbol>` | yes / no |
| `nervx ask returns-type <symbol>` | return type from signature |
| `nervx ask callers-count <symbol>` | integer |
| `nervx ask has-tests <symbol>` | yes / no + count |
| `nervx verify "A calls B"` | confirms or denies a call path (up to 6 hops) |

### Analysis
| Command | What it does |
|---------|-------------|
| `nervx callers <symbol>` | Who calls this function |
| `nervx blast-radius <symbol>` | Full downstream impact (before refactors) |
| `nervx trace <from> <to>` | Shortest call path (falls back to inheritance + dynamic dispatch) |
| `nervx find --dead` | Unreferenced code (framework-aware) |
| `nervx find --no-tests --importance-gt 20` | Critical untested code |
| `nervx find --exclude-category vendor,test` | Drop noise paths (vendor/test/generated/example/doc) |
| `nervx flows [keyword]` | End-to-end execution paths |
| `nervx diff --days 7` | Recent structural changes |
| `nervx cochange <file> --why` | Co-modified files with the commit hashes behind each coupling |
| `nervx string-refs <identifier>` | Every file:line where a string literal appears (all languages) |

### Diagnostics
| Command | What it does |
|---------|-------------|
| `nervx doctor` | Self-check: brain age, staleness, schema sanity, `.gitignore` coverage |
| `nervx nav "<q>" --verbose-warnings` | Show warning provenance (methodology, confidence, evidence) |

### Testing
| Command | What it does |
|---------|-------------|
| `nervx run pytest [args]` | Structured summary (~80 tokens vs 8000) |
| `nervx run pytest --raw <run_id>` | Retrieve cached raw output |

### Visualization & Stats
| Command | What it does |
|---------|-------------|
| `nervx viz .` | Interactive D3 visualization |
| `nervx stats` | Graph statistics |

All output commands support `--json` for machine-parseable output.

## Excluding Files

Create a `.nervxignore` in the repo root (gitignore syntax) to exclude files from indexing. Defaults already skip `__pycache__/`, `node_modules/`, `dist/`, `build/`, `.venv/`, minified bundles, lockfiles, and vendor directories.

## Claude Code Integration

When you run `nervx build`, it automatically adds instructions to your project's `CLAUDE.md` that teach Claude Code to use nervx commands. Claude will use `nervx nav` before exploring code, `nervx peek`/`nervx tree` instead of reading full files, `nervx ask`/`nervx verify` for quick verification, and `nervx blast-radius` before refactoring — saving tokens and tool calls.

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
