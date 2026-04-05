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
## nervx — Codebase Intelligence

This project has a nervx brain (`.nervx/brain.db`). **Use nervx commands before falling back to Grep/Glob/Read** — they return pre-indexed results in ~400 tokens instead of multi-step exploration costing thousands.

### Commands (run via Bash tool)

| Command | When to use | Example |
|---------|-------------|---------|
| `nervx nav "<question>"` | Before exploring code for any task | `nervx nav "how does auth work"` |
| `nervx blast-radius "<symbol_id>"` | Before refactoring a function/class | `nervx blast-radius "src/api.py::handle_request"` |
| `nervx find --dead` | Finding unreferenced dead code | `nervx find --dead --kind function` |
| `nervx find --no-tests --importance-gt 20` | Finding untested critical code | `nervx find --kind function --no-tests` |
| `nervx flows <keyword>` | Tracing execution paths | `nervx flows auth` |
| `nervx diff --days 7` | Seeing recent structural changes | `nervx diff --days 30` |

### What navigate returns

`nervx nav` returns ranked symbols, **execution flows** (call chains traced from matches), connected symbols, suggested read order, and warnings — all in one query.

### Workflow

1. **Start of session**: Read `NERVX.md` for project overview (module map, entry points, patterns, fragile zones)
2. **Before any code exploration**: Run `nervx nav "<your question>"` first — it returns the right files, line ranges, execution flows, read order, and warnings
3. **Before refactoring**: Run `nervx blast-radius "<symbol>"` to see all downstream callers (saves multiple rounds of grep)
4. **Before cleanup**: Run `nervx find --dead` to find unreferenced symbols that may be safe to remove
5. **Only then** fall back to Grep/Read for details nervx didn't cover

### Symbol ID format
Symbol IDs use the format `file_path::ClassName.method_name` or `file_path::function_name`. Example: `server/main.py::handle_request`
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
