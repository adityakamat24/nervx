"""NERVX.md briefing generator."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from nervx.memory.store import GraphStore


# Known framework indicators
FRAMEWORK_INDICATORS = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "asyncio": "asyncio",
    "websockets": "WebSockets",
    "react": "React",
    "express": "Express",
    "sqlalchemy": "SQLAlchemy",
    "pydantic": "Pydantic",
    "pytest": "pytest",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
}


def generate_briefing(store: GraphStore, repo_root: str) -> str:
    """Generate NERVX.md content from graph data."""
    repo_name = Path(repo_root).name
    tech_stack = _detect_tech_stack(store)
    node_count = store.get_meta("node_count") or "0"
    edge_count = store.get_meta("edge_count") or "0"

    lines = [
        f"# {repo_name}",
        tech_stack,
        f"_Graph: {node_count} symbols, {edge_count} edges_",
        "",
    ]

    # Module Map (cap at 15, collapse small modules)
    module_map = _build_module_map(store)
    if module_map:
        lines.append("## Module Map")
        shown = module_map[:15]
        for d, desc in shown:
            lines.append(f"  {d}/  {desc}")
        if len(module_map) > 15:
            lines.append(f"  ... and {len(module_map) - 15} more modules")
        lines.append("")

    # Entry Points (cap at 10, sorted by importance)
    entry_points = _find_entry_points(store)
    if entry_points:
        entry_points.sort(key=lambda n: -(n["importance"] or 0))
        lines.append("## Entry Points")
        for ep in entry_points[:10]:
            lines.append(f"  {ep['file_path']}::{ep['signature'] or ep['name']}")
        if len(entry_points) > 10:
            lines.append(f"  ... and {len(entry_points) - 10} more entry points")
        lines.append("")

    # Key Flows (cap at 5)
    paths = store.get_concept_paths()
    if paths:
        lines.append("## Key Flows")
        for p in paths[:5]:
            node_ids = json.loads(p["node_ids"]) if isinstance(p["node_ids"], str) else p["node_ids"]
            chain = " → ".join(
                _short_name(nid) for nid in node_ids[:6]
            )
            if len(node_ids) > 6:
                chain += " → ..."
            lines.append(f"  {p['name']}: {chain}")
        lines.append("")

    # Detected Patterns (cap at 10, grouped by type for conciseness)
    patterns = store.get_all_patterns()
    if patterns:
        lines.append("## Detected Patterns")
        # Group by pattern type, show count + top examples
        by_type: dict[str, list] = defaultdict(list)
        for p in patterns:
            by_type[p["pattern"]].append(p)
        for ptype in sorted(by_type.keys()):
            items = by_type[ptype]
            if len(items) <= 3:
                for p in items:
                    node = store.get_node(p["node_id"])
                    loc = node["name"] if node else p["node_id"]
                    lines.append(f"  {ptype.upper()}: {loc} → {p['implication']}")
            else:
                # Show count + top 2 examples
                lines.append(f"  {ptype.upper()} ({len(items)} instances):")
                for p in items[:2]:
                    node = store.get_node(p["node_id"])
                    loc = node["name"] if node else p["node_id"]
                    lines.append(f"    {loc} → {p['implication']}")
                lines.append(f"    ... and {len(items) - 2} more")
        lines.append("")

    # Hotspots (cap at 5)
    hotspots = _find_hotspots(store)
    if hotspots:
        lines.append("## Hotspots (last 30 days)")
        for h in hotspots[:5]:
            lines.append(
                f"  {h['file_path']}  {h['commits_30d']} commits "
                f"(by {h['primary_author']}, {h['author_count']} authors)"
            )
        if len(hotspots) > 5:
            lines.append(f"  ... and {len(hotspots) - 5} more active files")
        lines.append("")

    # Fragile Zones (cap at 5)
    fragile = _find_fragile_zones(store)
    if fragile:
        lines.append("## Fragile Zones")
        for f_node, reason in fragile[:5]:
            lines.append(
                f"  {f_node['file_path']}::{f_node['name']}  "
                f"importance={f_node['importance']:.1f}, {reason}"
            )
        lines.append("")

    # Temporal Couplings (cap at 5)
    couplings = _find_temporal_couplings(store)
    if couplings:
        lines.append("## Temporal Couplings")
        for c in couplings[:5]:
            lines.append(
                f"  {c['file_a']} <-> {c['file_b']}  "
                f"({int(c['coupling_score'] * 100)}%, {c['co_commit_count']} co-commits)"
            )
        lines.append("")

    return "\n".join(lines)


# ── CLAUDE.md integration ─────────────────────────────────────────

_CLAUDE_MD_START = "<!-- nervx:start -->"
_CLAUDE_MD_END = "<!-- nervx:end -->"


def generate_claude_instructions() -> str:
    """Generate the nervx section for CLAUDE.md."""
    return f"""{_CLAUDE_MD_START}
## nervx — codebase brain (auto-generated, do not edit this section)

nervx has pre-indexed this codebase into `.nervx/brain.db`. Use these commands
**before** falling back to grep/cat/Read — they return pre-computed answers in
tens to hundreds of tokens instead of thousands.

### EXPLORATION (use these BEFORE reading files)

| Command | What you get |
|---------|--------------|
| `nervx nav "<question>"` | ranked file:line results, call flows, read order, warnings |
| `nervx tree <file>` | structural overview of a file, ~150 tokens vs 4000 |
| `nervx peek <symbol>` | 50-token preview — signature, callees, caller count, test coverage, no source |

### READING (use these INSTEAD of cat/Read)

| Command | What you get |
|---------|--------------|
| `nervx read <symbol>` | source of one function/method |
| `nervx read <symbol> --context 1` | source of the symbol + everything it calls |
| `nervx read <symbol> --since <hash>` | returns "unchanged" (1 token) if the symbol hasn't been edited |

### QUICK ANSWERS (5–30 tokens each — use instead of reading source to verify)

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

### ANALYSIS

| Command | When to use |
|---------|-------------|
| `nervx callers <symbol>` | who calls this function (focused) |
| `nervx blast-radius <symbol>` | full downstream impact (before refactors) |
| `nervx trace <from> <to>` | shortest call path between two symbols; add `--read` for source |
| `nervx find --dead` | unreferenced code (framework-aware) |
| `nervx find --no-tests --importance-gt 20` | critical untested code |
| `nervx flows [keyword]` | end-to-end execution paths |
| `nervx diff --days 7` | recent structural changes |

### TESTING

| Command | What you get |
|---------|--------------|
| `nervx run pytest [args]` | structured summary (~80 tokens vs 8000 of traceback) |
| `nervx run pytest --raw <run_id>` | retrieve the full cached raw output |

### CROSS-LANGUAGE

| Command | What you get |
|---------|--------------|
| `nervx string-refs <identifier>` | every file:line where this string literal appears, across all languages |

### WORKFLOW

1. Start with `nervx tree` / `nervx peek` to explore — NOT cat/Read.
2. Use `nervx ask` / `nervx verify` for quick verification — NOT reading source.
3. Use `nervx read --context 1` for targeted reading — NOT full file reads.
4. Use `nervx run pytest` for test results — NOT raw pytest output.
5. If nervx commands fail or return nothing useful, then fall back to grep/cat.

### Symbol ID format
`file_path::ClassName.method_name` or `file_path::function_name`. Example:
`server/main.py::handle_request`. Fuzzy matching is built in — short names like
`handle_request` usually resolve automatically, and ambiguous queries return a
"did you mean?" list.

### All commands support `--json`
Every output command accepts `--json` for machine-parseable output.

### Excluding files
Create a `.nervxignore` in the repo root (gitignore syntax) to exclude files.
Defaults already skip `__pycache__/`, `node_modules/`, `dist/`, `build/`,
`.venv/`, minified bundles, lockfiles, vendor dirs, etc.

NERVX.md contains the full architectural overview of this project.
{_CLAUDE_MD_END}"""


def inject_claude_md(repo_root: str) -> bool:
    """Add or update the nervx section in the project's CLAUDE.md.

    Returns True if CLAUDE.md was modified.
    """
    claude_md_path = os.path.join(repo_root, "CLAUDE.md")
    new_section = generate_claude_instructions()

    if os.path.exists(claude_md_path):
        with open(claude_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if nervx section already exists
        if _CLAUDE_MD_START in content:
            # Replace existing section
            import re
            pattern = re.escape(_CLAUDE_MD_START) + r".*?" + re.escape(_CLAUDE_MD_END)
            updated = re.sub(pattern, new_section, content, flags=re.DOTALL)
            if updated != content:
                with open(claude_md_path, "w", encoding="utf-8") as f:
                    f.write(updated)
                return True
            return False
        else:
            # Append to existing file
            with open(claude_md_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_section + "\n")
            return True
    else:
        # Create new CLAUDE.md
        with open(claude_md_path, "w", encoding="utf-8") as f:
            f.write(new_section + "\n")
        return True


def _detect_tech_stack(store: GraphStore) -> str:
    """Detect tech stack from imports and file patterns."""
    detected = set()
    # Check Python
    detected.add("Python")

    # Check imports
    nodes = store.get_all_nodes()
    all_names = set()
    for n in nodes:
        all_names.add(n["name"].lower())
        if n["signature"]:
            all_names.add(n["signature"].lower())

    # Check keywords
    for framework, label in FRAMEWORK_INDICATORS.items():
        results = store.search_keywords([framework])
        if results:
            detected.add(label)

    if detected:
        return ", ".join(sorted(detected))
    return "Python"


def _build_module_map(store: GraphStore) -> list[tuple[str, str]]:
    """Build module map with auto-descriptions."""
    # Group nodes by top-level directory
    dir_nodes: dict[str, list[dict]] = defaultdict(list)

    for node in store.get_all_nodes():
        fp = node["file_path"]
        if "/" in fp:
            top_dir = fp.split("/")[0]
        else:
            continue
        dir_nodes[top_dir].append(node)

    result = []
    for d in sorted(dir_nodes.keys()):
        nodes = dir_nodes[d]
        desc = _describe_module(nodes)
        result.append((d, desc))

    return result


def _describe_module(nodes: list[dict]) -> str:
    """Auto-describe a module based on dominant tags and kinds."""
    tag_counts: Counter = Counter()
    kind_counts: Counter = Counter()

    for n in nodes:
        kind_counts[n["kind"]] += 1
        tags = json.loads(n["tags"]) if isinstance(n["tags"], str) else n["tags"]
        for t in tags:
            if t.startswith("extends:"):
                continue
            tag_counts[t] += 1

    # Check dominant tags
    total = len(nodes)
    if tag_counts.get("test", 0) > total * 0.3:
        return "tests"
    if tag_counts.get("route_handler", 0) > total * 0.3:
        return "API routes/handlers"
    if tag_counts.get("data_model", 0) > total * 0.3:
        return "data models"
    if tag_counts.get("callback", 0) > total * 0.3:
        return "event handlers"
    if tag_counts.get("factory", 0) > total * 0.2:
        return "factories"

    # Default: count classes and functions
    n_classes = kind_counts.get("class", 0)
    n_funcs = kind_counts.get("function", 0) + kind_counts.get("method", 0)
    parts = []
    if n_classes:
        parts.append(f"{n_classes} classes")
    if n_funcs:
        parts.append(f"{n_funcs} functions")
    return ", ".join(parts) if parts else "module"


def _find_entry_points(store: GraphStore) -> list[dict]:
    """Find entrypoint-tagged nodes."""
    result = []
    for node in store.get_all_nodes():
        tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]
        if "entrypoint" in tags or "route_handler" in tags:
            result.append(node)
    return result


def _find_hotspots(store: GraphStore) -> list[dict]:
    """Find files with high recent commit activity."""
    stats = store.get_all_file_stats()
    hotspots = [s for s in stats if s["commits_30d"] > 0]
    hotspots.sort(key=lambda s: -s["commits_30d"])
    return hotspots


def _find_fragile_zones(store: GraphStore) -> list[tuple[dict, str]]:
    """Find important nodes with hazards.

    Thresholds:
    - importance >= 25 with any warning qualifies
    - importance >= 15 needs a serious warning (many callers, contract conflict)
      or multiple warning types
    """
    conflict_ids = set(store.get_contract_conflicts())
    result = []
    for node in store.get_all_nodes():
        importance = node["importance"] or 0.0
        if importance < 15:
            continue
        if node["kind"] == "file":
            continue

        reasons = []
        has_serious = False

        # No test coverage
        tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]
        if "test" not in tags:
            has_test = False
            edges = store.get_edges_from(node["id"]) + store.get_edges_to(node["id"])
            for e in edges:
                other_id = e["target_id"] if e["source_id"] == node["id"] else e["source_id"]
                other = store.get_node(other_id)
                if other:
                    otags = json.loads(other["tags"]) if isinstance(other["tags"], str) else other["tags"]
                    if "test" in otags:
                        has_test = True
                        break
            if not has_test:
                reasons.append("no tests")

        # Many callers
        called_by = [e for e in store.get_edges_from(node["id"])
                     if e["edge_type"] == "called_by"]
        if len(called_by) >= 8:
            reasons.append(f"{len(called_by)} callers")
            has_serious = True

        # Contract conflicts
        if node["id"] in conflict_ids:
            reasons.append("contract conflict")
            has_serious = True

        if not reasons:
            continue

        # Gate: importance >= 25 always qualifies; 15-25 needs serious or multiple
        if importance < 25 and not has_serious and len(reasons) < 2:
            continue

        result.append((node, ", ".join(reasons)))

    result.sort(key=lambda x: -x[0]["importance"])
    return result


def _find_temporal_couplings(store: GraphStore) -> list[dict]:
    """Find high-coupling file pairs."""
    rows = store.conn.execute(
        "SELECT * FROM cochanges WHERE coupling_score >= 0.4 ORDER BY coupling_score DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _short_name(node_id: str) -> str:
    """Extract short name from a node ID."""
    if "::" in node_id:
        return node_id.split("::")[-1]
    return node_id.split("/")[-1] if "/" in node_id else node_id
