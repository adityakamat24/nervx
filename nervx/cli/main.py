"""CLI argument parsing and dispatch for nervx."""

from __future__ import annotations

import argparse
import io
import os
import sys

from nervx.memory.store import GraphStore

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _resolve_repo(args) -> str:
    """Get the repo root from args or cwd."""
    return os.path.abspath(getattr(args, "repo", None) or getattr(args, "path", None) or os.getcwd())


def _db_path(repo_root: str) -> str:
    """Get the brain.db path for a repo."""
    return os.path.join(repo_root, ".nervx", "brain.db")


def _ensure_brain(repo_root: str) -> str:
    """Check that brain.db exists, exit with error if not."""
    db = _db_path(repo_root)
    if not os.path.exists(db):
        print("No brain found. Run 'nervx build' first.", file=sys.stderr)
        sys.exit(1)
    return db


def cmd_build(args):
    """Full build command."""
    from nervx.build import full_build

    repo_root = os.path.abspath(args.path or os.getcwd())
    nervx_dir = os.path.join(repo_root, ".nervx")
    os.makedirs(nervx_dir, exist_ok=True)

    # Create .gitignore in .nervx/
    gitignore = os.path.join(nervx_dir, ".gitignore")
    if not os.path.exists(gitignore):
        with open(gitignore, "w") as f:
            f.write("*\n")

    db = os.path.join(nervx_dir, "brain.db")
    full_build(repo_root, db)

    # Generate NERVX.md if briefing module is available
    try:
        from nervx.attention.briefing import generate_briefing, inject_claude_md
        store = GraphStore(db)
        briefing = generate_briefing(store, repo_root)
        store.close()
        nervx_md = os.path.join(repo_root, "NERVX.md")
        with open(nervx_md, "w", encoding="utf-8") as f:
            f.write(briefing)
        print(f"Generated {nervx_md}")

        # Add/update nervx section in CLAUDE.md
        if inject_claude_md(repo_root):
            print(f"Updated {os.path.join(repo_root, 'CLAUDE.md')} with nervx instructions")
    except ImportError:
        pass


def cmd_update(args):
    """Incremental update command."""
    from nervx.build import incremental_update

    repo_root = os.path.abspath(args.path or os.getcwd())
    db = _ensure_brain(repo_root)
    incremental_update(repo_root, db)

    try:
        from nervx.attention.briefing import generate_briefing, inject_claude_md
        store = GraphStore(db)
        briefing = generate_briefing(store, repo_root)
        store.close()
        nervx_md = os.path.join(repo_root, "NERVX.md")
        with open(nervx_md, "w", encoding="utf-8") as f:
            f.write(briefing)

        inject_claude_md(repo_root)
    except ImportError:
        pass


def cmd_navigate(args):
    """Navigate query command."""
    from nervx.attention.query import navigate

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = navigate(store, args.query, budget=args.budget)
    print(result.formatted)
    store.close()


def cmd_blast_radius(args):
    """Blast radius command."""
    from nervx.attention.query import blast_radius_query

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    output = blast_radius_query(store, args.symbol, depth=args.depth)
    print(output)
    store.close()


def cmd_callers(args):
    """Callers command — show what calls a symbol."""
    from nervx.attention.callers import find_callers

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    output = find_callers(store, args.symbol, max_depth=args.depth)
    print(output)
    store.close()


def cmd_read(args):
    """Read command — print source of a symbol (plus optional callees)."""
    from nervx.attention.reader import read_symbol

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    depth = min(max(0, args.context), 3)
    output = read_symbol(store, args.symbol, context_depth=depth, repo_root=repo_root)
    print(output)
    store.close()


def cmd_flows(args):
    """Flows command."""
    import json

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    paths = store.get_concept_paths()

    keyword = args.keyword
    for p in paths:
        name = p["name"]
        if keyword and keyword.lower() not in name.lower():
            continue
        node_ids = json.loads(p["node_ids"]) if isinstance(p["node_ids"], str) else p["node_ids"]
        chain = " -> ".join(nid.split("::")[-1] if "::" in nid else nid for nid in node_ids)
        print(f"  {name}: {chain}  [{p['path_type']}]")

    store.close()


def cmd_find(args):
    """Structural find command."""
    from nervx.attention.query import find

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    results = find(
        store,
        kind=args.kind,
        tag=args.tag,
        no_tests=args.no_tests,
        importance_gt=args.importance_gt,
        cross_module=args.cross_module,
        dead=getattr(args, "dead", False),
    )
    for node in results:
        line = f"  {node['file_path']}:{node['line_start']}  {node['name']}  [{node['kind']}]"
        if node["importance"] > 0:
            line += f"  importance={node['importance']:.1f}"
        print(line)
    store.close()


def cmd_diff(args):
    """Diff command."""
    from nervx.attention.query import diff_query

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    output = diff_query(store, days=args.days)
    print(output)
    store.close()


def cmd_stats(args):
    """Stats command."""
    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)

    print("nervx stats:")
    for key in ["file_count", "node_count", "edge_count",
                "build_time_seconds", "last_build", "last_update"]:
        val = store.get_meta(key)
        if val:
            print(f"  {key}: {val}")

    # Pattern counts
    patterns = store.get_all_patterns()
    if patterns:
        from collections import Counter
        pcounts = Counter(p["pattern"] for p in patterns)
        print("  patterns:")
        for pat, cnt in pcounts.most_common():
            print(f"    {pat}: {cnt}")

    store.close()


def cmd_briefing(args):
    """Regenerate NERVX.md."""
    from nervx.attention.briefing import generate_briefing

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    briefing = generate_briefing(store, repo_root)
    store.close()

    nervx_md = os.path.join(repo_root, "NERVX.md")
    with open(nervx_md, "w", encoding="utf-8") as f:
        f.write(briefing)
    print(f"Generated {nervx_md}")


def cmd_viz(args):
    """Export data and open interactive visualization."""
    import shutil
    from pathlib import Path
    from nervx.viz.export import export_viz_data, write_viz_json
    from nervx.viz.server import serve_viz

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)

    max_nodes = getattr(args, "max_nodes", 1000)
    data = export_viz_data(store, max_nodes=max_nodes)
    store.close()

    if data["meta"].get("truncated"):
        print(f"Large repo: showing top {data['meta']['viz_node_count']} of "
              f"{data['meta']['total_node_count']} nodes by importance")

    nervx_dir = os.path.join(repo_root, ".nervx")
    os.makedirs(nervx_dir, exist_ok=True)

    json_path = os.path.join(nervx_dir, "nervx-viz.json")
    write_viz_json(data, json_path)
    print(f"Exported {json_path}")

    if getattr(args, "export_only", False):
        return

    # Copy template HTML
    template = Path(__file__).parent.parent / "viz" / "template.html"
    index_path = os.path.join(nervx_dir, "index.html")
    shutil.copy2(str(template), index_path)

    port = getattr(args, "port", 8741)
    serve_viz(nervx_dir, port=port)


def cmd_export(args):
    """Export brain.db to nervx-viz.json."""
    from nervx.viz.export import export_viz_data, write_viz_json

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)

    max_nodes = getattr(args, "max_nodes", 1000)
    data = export_viz_data(store, max_nodes=max_nodes)
    store.close()

    if data["meta"].get("truncated"):
        print(f"Large repo: showing top {data['meta']['viz_node_count']} of "
              f"{data['meta']['total_node_count']} nodes by importance")

    output = getattr(args, "output", None) or os.path.join(repo_root, ".nervx", "nervx-viz.json")
    write_viz_json(data, output)
    print(f"Exported {output}")


def cmd_watch(args):
    """Watch for file changes and auto-update."""
    from nervx.cli.watch import run_watch

    repo_root = os.path.abspath(args.path or os.getcwd())
    db = _ensure_brain(repo_root)
    debounce = getattr(args, "debounce", 2.0)
    run_watch(repo_root, db, debounce=debounce)


def main():
    parser = argparse.ArgumentParser(
        prog="nervx",
        description="A codebase brain for Claude",
    )
    subparsers = parser.add_subparsers(dest="command")

    # build
    p_build = subparsers.add_parser("build", help="Full build")
    p_build.add_argument("path", nargs="?", default=None, help="Repository path")
    p_build.set_defaults(func=cmd_build)

    # update
    p_update = subparsers.add_parser("update", help="Incremental update")
    p_update.add_argument("path", nargs="?", default=None, help="Repository path")
    p_update.set_defaults(func=cmd_update)

    # navigate
    p_nav = subparsers.add_parser("navigate", aliases=["nav", "n"], help="Navigate query")
    p_nav.add_argument("query", help="Natural language query")
    p_nav.add_argument("--budget", type=int, default=5, help="Max primary results")
    p_nav.add_argument("--repo", default=None, help="Repository path")
    p_nav.set_defaults(func=cmd_navigate)

    # blast-radius
    p_blast = subparsers.add_parser("blast-radius", aliases=["blast", "br"],
                                     help="Compute blast radius")
    p_blast.add_argument("symbol", help="Symbol ID")
    p_blast.add_argument("--depth", type=int, default=3, help="Max depth")
    p_blast.add_argument("--repo", default=None, help="Repository path")
    p_blast.set_defaults(func=cmd_blast_radius)

    # callers
    p_callers = subparsers.add_parser("callers", help="Show what calls a symbol")
    p_callers.add_argument("symbol", help="Symbol ID")
    p_callers.add_argument("--depth", type=int, default=1,
                           help="Caller depth (default: 1)")
    p_callers.add_argument("--repo", default=None, help="Repository path")
    p_callers.set_defaults(func=cmd_callers)

    # read
    p_read = subparsers.add_parser("read", help="Read source code of a symbol")
    p_read.add_argument("symbol", help="Symbol ID to read")
    p_read.add_argument("--context", type=int, default=0,
                        help="Include callees up to this depth (default: 0, max: 3)")
    p_read.add_argument("--repo", default=None, help="Repository path")
    p_read.set_defaults(func=cmd_read)

    # flows
    p_flows = subparsers.add_parser("flows", help="Show concept paths")
    p_flows.add_argument("keyword", nargs="?", default=None, help="Filter keyword")
    p_flows.add_argument("--repo", default=None, help="Repository path")
    p_flows.set_defaults(func=cmd_flows)

    # find
    p_find = subparsers.add_parser("find", help="Structural query")
    p_find.add_argument("--kind", choices=["function", "class", "method", "file"])
    p_find.add_argument("--tag", help="Filter by tag")
    p_find.add_argument("--no-tests", action="store_true", help="Only symbols with no test coverage")
    p_find.add_argument("--importance-gt", type=float, help="Minimum importance")
    p_find.add_argument("--cross-module", action="store_true", help="Only cross-module symbols")
    p_find.add_argument("--dead", action="store_true", help="Only unreferenced symbols (dead code)")
    p_find.add_argument("--repo", default=None, help="Repository path")
    p_find.set_defaults(func=cmd_find)

    # diff
    p_diff = subparsers.add_parser("diff", help="Show structural changes")
    p_diff.add_argument("--days", type=int, default=7, help="Days to look back")
    p_diff.add_argument("--repo", default=None, help="Repository path")
    p_diff.set_defaults(func=cmd_diff)

    # stats
    p_stats = subparsers.add_parser("stats", help="Print graph statistics")
    p_stats.add_argument("--repo", default=None, help="Repository path")
    p_stats.set_defaults(func=cmd_stats)

    # briefing
    p_brief = subparsers.add_parser("briefing", help="Regenerate NERVX.md")
    p_brief.add_argument("--repo", default=None, help="Repository path")
    p_brief.set_defaults(func=cmd_briefing)

    # viz
    p_viz = subparsers.add_parser("viz", help="Open interactive visualization")
    p_viz.add_argument("path", nargs="?", default=None, help="Repository path")
    p_viz.add_argument("--port", type=int, default=8741, help="Server port")
    p_viz.add_argument("--export-only", action="store_true", help="Only export JSON")
    p_viz.add_argument("--max-nodes", type=int, default=1000, help="Max nodes to export (0=all)")
    p_viz.add_argument("--repo", default=None, help="Repository path")
    p_viz.set_defaults(func=cmd_viz)

    # export
    p_export = subparsers.add_parser("export", help="Export brain to JSON")
    p_export.add_argument("output", nargs="?", default=None, help="Output path")
    p_export.add_argument("--max-nodes", type=int, default=1000, help="Max nodes to export (0=all)")
    p_export.add_argument("--repo", default=None, help="Repository path")
    p_export.set_defaults(func=cmd_export)

    # watch (only available if watchdog is installed)
    try:
        import watchdog  # noqa: F401
        p_watch = subparsers.add_parser("watch", help="Watch for changes and auto-update")
        p_watch.add_argument("path", nargs="?", default=None, help="Repository path")
        p_watch.add_argument("--debounce", type=float, default=2.0,
                             help="Seconds to wait before triggering update (default: 2)")
        p_watch.set_defaults(func=cmd_watch)
    except ImportError:
        pass

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
