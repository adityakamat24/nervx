"""`nervx trace <from> <to>` — shortest call-path between two symbols."""

from __future__ import annotations

from pathlib import Path

from nervx.attention.fuzzy import resolve_symbol
from nervx.attention.graph_paths import bfs_path
from nervx.memory.store import GraphStore


def trace_path(
    store: GraphStore,
    source_symbol: str,
    target_symbol: str,
    include_source: bool = False,
    repo_root: str = ".",
    calls_only: bool = False,
    via_inheritance: bool = False,
    pick_source: int | None = None,
    pick_target: int | None = None,
) -> dict | str:
    """Compute the shortest call path from source to target.

    By default tries a strict ``calls``-only BFS first, and if that fails,
    falls back to a second BFS that also walks ``inherits``/``inherited_by``
    plus ``dispatches_to``/``dispatched_from`` edges — this catches the
    "polymorphic dispatch through an abstract base" case that static
    call-linking can't resolve. The fallback path is labelled as *soft* in
    the output so callers don't treat it as a confirmed runtime path.

    ``calls_only``      — disable the inheritance/dispatch fallback entirely.
    ``via_inheritance`` — skip straight to the inheritance-aware BFS.
    """
    src, err_s = resolve_symbol(store, source_symbol, pick=pick_source)
    if src is None:
        return err_s
    dst, err_t = resolve_symbol(store, target_symbol, pick=pick_target)
    if dst is None:
        return err_t

    path: list[str] = []
    resolution: str = "calls"  # "calls" (strict) | "inheritance" (soft) | ""

    if not via_inheritance:
        path = bfs_path(store, src["id"], dst["id"], edge_type="calls", max_depth=8)
        if path:
            resolution = "calls"

    if not path and not calls_only:
        # Fallback: include inheritance + dispatches_to edges. This does NOT
        # guarantee a runtime path; it's evidence that source reaches target
        # through a base class (and possibly a polymorphic override) in the
        # static graph.
        path = bfs_path(
            store,
            src["id"],
            dst["id"],
            edge_type=(
                "calls", "inherits", "inherited_by",
                "dispatches_to", "dispatched_from",
            ),
            max_depth=8,
        )
        if path:
            resolution = "inheritance"

    if not path:
        return {
            "found": False,
            "source": src["id"],
            "target": dst["id"],
            "hops": [],
            "length": 0,
            "resolution": "",
            "note": (
                f"No static call path from {src['id']} to {dst['id']} "
                f"(searched 8 hops on calls edges"
                f"{'' if calls_only else ' + inheritance fallback'}).\n"
                f"Note: nervx uses static name-resolution — polymorphic "
                f"dispatch through abstract base classes may not be "
                f"captured. For call-site evidence, try: "
                f"nervx callers {dst['id'].split('::')[-1]}"
            ),
        }

    hops: list[dict] = []
    for nid in path:
        n = store.get_node(nid)
        if n is None:
            continue
        hop = {
            "id": nid,
            "name": n["name"],
            "file_path": n["file_path"],
            "line_start": n.get("line_start") or 0,
            "line_end": n.get("line_end") or 0,
            "signature": (n.get("signature") or "").strip(),
        }
        if include_source:
            hop["source"] = _read_source(n, repo_root) or ""
        hops.append(hop)

    return {
        "found": True,
        "source": src["id"],
        "target": dst["id"],
        "hops": hops,
        "length": len(hops),
        "resolution": resolution,
    }


def format_trace(result: dict | str) -> str:
    if isinstance(result, str):
        return result
    if not result["found"]:
        return result.get("note") or (
            f"No call path from {result['source']} to {result['target']}."
        )

    resolution = result.get("resolution", "calls")
    suffix = "" if resolution == "calls" else ", via inheritance"
    header = (
        f"## trace: {result['source']} → {result['target']}  "
        f"({result['length']} hops{suffix})"
    )
    lines: list[str] = [header, ""]
    if resolution == "inheritance":
        lines.append(
            "  ⚠ No direct static call path found. This is a SOFT path that "
            "crosses inheritance edges — resolution depends on runtime "
            "polymorphic dispatch and may not reflect the actual execution "
            "path. Treat as evidence, not proof."
        )
        lines.append("")
    for i, hop in enumerate(result["hops"]):
        arrow = "  " if i == 0 else "  ↓ "
        loc = f"{hop['file_path']}:{hop['line_start']}"
        sig = hop["signature"] or hop["name"]
        lines.append(f"{arrow}{loc}  {sig}")
        if "source" in hop and hop["source"]:
            lines.append("")
            lines.append(hop["source"])
            lines.append("")
    return "\n".join(lines)


def _read_source(node: dict, repo_root: str) -> str | None:
    fp = node.get("file_path")
    if not fp:
        return None
    full = Path(repo_root) / fp
    if not full.exists():
        return None
    try:
        text = full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    ls = node.get("line_start") or 0
    le = node.get("line_end") or 0
    if not ls or not le:
        return None
    return "\n".join(lines[max(0, ls - 1):le])
