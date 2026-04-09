"""CLI argument parsing and dispatch for nervx."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from nervx.memory.store import GraphStore


def _emit(args, text: str, data) -> None:
    """Print either JSON or the pre-formatted text, based on ``--json``."""
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(text)

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
    """Check that brain.db exists, exit with error if not.

    Also runs a cheap staleness probe (B13) and prints a stderr hint if the
    on-disk source tree has moved on from the indexed snapshot. The probe
    is bounded so it never adds noticeable latency to a query.
    """
    db = _db_path(repo_root)
    if not os.path.exists(db):
        print("No brain found. Run 'nervx build' first.", file=sys.stderr)
        sys.exit(1)
    _warn_if_stale(repo_root, db)
    return db


def _warn_if_stale(repo_root: str, db_path: str) -> None:
    """Print a one-liner to stderr if enough source files are newer than
    the last build. Bounded walk (~500 files) to keep the overhead under
    the threshold of perceptible latency on a cold cache.
    """
    import time
    from datetime import datetime

    try:
        store = GraphStore(db_path)
        last_build_iso = store.get_meta("last_build") or store.get_meta("last_update") or ""
        store.close()
    except Exception:
        return
    if not last_build_iso:
        return
    try:
        # Parse ISO 8601 with or without trailing Z.
        cleaned = last_build_iso.replace("Z", "+00:00")
        last_build_ts = datetime.fromisoformat(cleaned).timestamp()
    except (TypeError, ValueError):
        return

    try:
        from nervx.build import walk_files
    except Exception:
        return

    try:
        files = walk_files(repo_root)
    except Exception:
        return

    if not files:
        return

    # Bounded scan: first ~500 files is enough to detect "I've been editing".
    sample_size = min(len(files), 500)
    newer = 0
    start = time.monotonic()
    for fp in files[:sample_size]:
        if time.monotonic() - start > 0.25:  # hard 250ms cap on the probe
            break
        try:
            if os.path.getmtime(fp) > last_build_ts + 1:
                newer += 1
        except OSError:
            continue

    if newer == 0:
        return
    # Fire if more than 5 files OR more than 10% of the scanned sample.
    if newer > 5 or newer > sample_size * 0.10:
        print(
            f"\u26a0 {newer} file(s) modified since last build "
            f"(scanned {sample_size}). Run 'nervx update' for fresh results.",
            file=sys.stderr,
        )


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
    from dataclasses import asdict
    from nervx.attention.query import navigate

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = navigate(
        store,
        args.query,
        budget=args.budget,
        exclude_category=getattr(args, "exclude_category", None),
        include_category=getattr(args, "include_category", None),
    )
    if getattr(args, "json", False):
        payload = {
            "query": args.query,
            "terms": result.terms,
            "primary": result.primary,
            "secondary": result.secondary,
            "cochange_files": result.cochange_files,
            "read_order": result.read_order,
            "warnings": [asdict(w) for w in result.warnings],
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        text = result.formatted
        if getattr(args, "verbose_warnings", False) and result.warnings:
            text += "\n### Warning Provenance\n"
            for w in result.warnings:
                text += (
                    f"  [{w.category}] confidence={w.confidence}\n"
                    f"    methodology: {w.methodology}\n"
                )
                if w.evidence:
                    text += f"    evidence: {', '.join(str(e) for e in w.evidence[:5])}\n"
        print(text)
    store.close()


def cmd_blast_radius(args):
    """Blast radius command."""
    from nervx.attention.query import blast_radius_query

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    output = blast_radius_query(
        store, args.symbol, depth=args.depth,
        pick=getattr(args, "pick", None),
        exclude_category=getattr(args, "exclude_category", None),
        include_category=getattr(args, "include_category", None),
    )
    if getattr(args, "json", False):
        print(json.dumps({"symbol": args.symbol, "output": output}, indent=2))
    else:
        print(output)
    store.close()


def cmd_callers(args):
    """Callers command — show what calls a symbol."""
    from nervx.attention.callers import find_callers

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    output = find_callers(
        store, args.symbol, max_depth=args.depth,
        pick=getattr(args, "pick", None),
    )
    if getattr(args, "json", False):
        print(json.dumps({"symbol": args.symbol, "output": output}, indent=2))
    else:
        print(output)
    store.close()


def cmd_read(args):
    """Read command — print source of a symbol (plus optional callees)."""
    from nervx.attention.reader import read_symbol

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    depth = min(max(0, args.context), 3)
    output = read_symbol(
        store,
        args.symbol,
        context_depth=depth,
        repo_root=repo_root,
        since_hash=getattr(args, "since", "") or "",
        pick=getattr(args, "pick", None),
    )
    if getattr(args, "json", False):
        print(json.dumps({"symbol": args.symbol, "source": output}, indent=2))
    else:
        print(output)
    store.close()


def cmd_peek(args):
    """Peek command — 50-token preview of a symbol."""
    from nervx.attention.peek import peek_symbol, format_peek

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = peek_symbol(
        store, args.symbol, repo_root=repo_root,
        pick=getattr(args, "pick", None),
    )
    _emit(args, format_peek(result), result)
    store.close()


def cmd_tree(args):
    """Tree command — structural overview of a file."""
    from nervx.attention.tree import tree_file, format_tree

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = tree_file(store, args.file, repo_root=repo_root)
    _emit(args, format_tree(result), result)
    store.close()


def cmd_verify(args):
    """Verify command — yes/no graph path check."""
    from nervx.attention.verify import verify_statement, format_verify

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = verify_statement(store, args.statement)
    _emit(args, format_verify(result), result)
    store.close()


def cmd_ask(args):
    """Ask family — micro-queries for structural questions."""
    from nervx.attention.ask import run_ask, format_ask

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = run_ask(
        store, args.subcommand, list(args.args or []),
        pick=getattr(args, "pick", None),
    )
    _emit(args, format_ask(result), result)
    store.close()


def cmd_trace(args):
    """Trace command — shortest call path between two symbols."""
    from nervx.attention.trace import trace_path, format_trace

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = trace_path(
        store,
        args.source,
        args.target,
        include_source=getattr(args, "read", False),
        repo_root=repo_root,
        calls_only=getattr(args, "calls_only", False),
        via_inheritance=getattr(args, "via_inheritance", False),
        pick_source=getattr(args, "pick_source", None),
        pick_target=getattr(args, "pick_target", None),
    )
    _emit(args, format_trace(result), result)
    store.close()


def cmd_run(args):
    """Run command — execute test runners and return compact summaries."""
    from nervx.tools.runners import run_pytest, read_raw

    repo_root = _resolve_repo(args)
    nervx_dir = os.path.join(repo_root, ".nervx")
    os.makedirs(nervx_dir, exist_ok=True)

    if args.runner != "pytest":
        print(f"Unknown runner: {args.runner}", file=sys.stderr)
        sys.exit(2)

    # argparse REMAINDER swallows everything after the runner positional,
    # including our own --raw flag, so pull it out of pytest_args manually.
    pytest_args = list(args.pytest_args or [])
    raw_id = getattr(args, "raw", "") or ""
    if "--raw" in pytest_args:
        idx = pytest_args.index("--raw")
        if idx + 1 < len(pytest_args):
            raw_id = pytest_args[idx + 1]
            del pytest_args[idx:idx + 2]
        else:
            del pytest_args[idx]

    if raw_id:
        print(read_raw(raw_id, nervx_dir))
        return

    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        output = run_pytest(pytest_args, nervx_dir)
    finally:
        os.chdir(original_cwd)

    if getattr(args, "json", False):
        print(json.dumps({"runner": "pytest", "output": output}, indent=2))
    else:
        print(output)


def cmd_string_refs(args):
    """String-refs command — cross-language literal lookup."""
    from nervx.attention.string_refs import find_string_refs, format_string_refs

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    result = find_string_refs(store, args.identifier)
    _emit(args, format_string_refs(result), result)
    store.close()


def cmd_flows(args):
    """Flows command."""
    import json

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    paths = store.get_concept_paths()

    keyword = args.keyword
    emitted: list[dict] = []
    for p in paths:
        name = p["name"]
        if keyword and keyword.lower() not in name.lower():
            continue
        node_ids = json.loads(p["node_ids"]) if isinstance(p["node_ids"], str) else p["node_ids"]
        # Papercut: when two consecutive steps share the same short name
        # (e.g. ``radix_tree_v2 -> radix_tree_v2``), prepend the file
        # basename so the chain communicates which file is being traversed
        # instead of looking like a self-loop. File-only nodes keep their
        # basename as-is.
        def _short(nid: str) -> str:
            if "::" in nid:
                tail = nid.split("::", 1)[1]
                return tail.rsplit(".", 1)[-1] if "." in tail else tail
            # File node: use basename
            return nid.rsplit("/", 1)[-1]

        def _file_base(nid: str) -> str:
            if "::" in nid:
                return nid.split("::", 1)[0].rsplit("/", 1)[-1]
            return ""  # file nodes already use the basename as short

        short_names = [_short(nid) for nid in node_ids]
        labels: list[str] = []
        for i, nid in enumerate(node_ids):
            short = short_names[i]
            file_base = _file_base(nid)
            prev_same = i > 0 and short_names[i - 1] == short
            next_same = i + 1 < len(short_names) and short_names[i + 1] == short
            if file_base and (prev_same or next_same):
                labels.append(f"{file_base}:{short}")
            else:
                labels.append(short)
        chain = " -> ".join(labels)
        line = f"  {name}: {chain}  [{p['path_type']}]"
        emitted.append({
            "name": name,
            "chain": labels,
            "path_type": p["path_type"],
            "line": line,
        })

    if getattr(args, "json", False):
        print(json.dumps(
            [{"name": e["name"], "chain": e["chain"], "path_type": e["path_type"]}
             for e in emitted],
            indent=2,
        ))
    else:
        for e in emitted:
            print(e["line"])

    store.close()


def cmd_find(args):
    """Structural find command."""
    from nervx.attention.query import find

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)

    # Default category exclusion for --no-tests / --no-direct-tests:
    # tautologically-untested code (fixtures, docs, examples, vendored, CI
    # scripts) otherwise drowns out the real "critical untested code" signal.
    # Skip the default if the user passed explicit category filters or the
    # --include-test-fixtures opt-out.
    exclude_category = getattr(args, "exclude_category", None)
    include_category = getattr(args, "include_category", None)
    if (
        (args.no_tests or getattr(args, "no_direct_tests", False))
        and not exclude_category
        and not include_category
        and not getattr(args, "include_test_fixtures", False)
    ):
        exclude_category = ["test", "doc", "example", "vendor", "generated", "script"]

    results = find(
        store,
        kind=args.kind,
        tag=args.tag,
        no_tests=args.no_tests,
        no_direct_tests=getattr(args, "no_direct_tests", False),
        importance_gt=args.importance_gt,
        cross_module=args.cross_module,
        dead=getattr(args, "dead", False),
        exclude_category=exclude_category,
        include_category=include_category,
    )
    if getattr(args, "json", False):
        print(json.dumps([dict(n) for n in results], indent=2, default=str))
    else:
        for node in results:
            line = f"  {node['file_path']}:{node['line_start']}  {node['name']}  [{node['kind']}]"
            if node["importance"] > 0:
                rank = node.get("importance_rank") or 0
                if rank > 0:
                    line += f"  importance={node['importance']:.1f} (rank {rank})"
                else:
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
    if getattr(args, "json", False):
        # diff_query currently returns a string blob; wrap it so JSON
        # consumers still get a stable shape they can key off.
        print(json.dumps({"days": args.days, "text": output}, indent=2))
    else:
        print(output)
    store.close()


def cmd_stats(args):
    """Stats command."""
    from collections import Counter

    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)

    meta: dict[str, str] = {}
    for key in ["file_count", "node_count", "edge_count",
                "build_time_seconds", "last_build", "last_update"]:
        val = store.get_meta(key)
        if val:
            meta[key] = val

    patterns = store.get_all_patterns()
    pcounts: dict[str, int] = {}
    if patterns:
        for pat, cnt in Counter(p["pattern"] for p in patterns).most_common():
            pcounts[pat] = cnt

    if getattr(args, "json", False):
        print(json.dumps({"meta": meta, "patterns": pcounts}, indent=2, default=str))
    else:
        print("nervx stats:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        if pcounts:
            print("  patterns:")
            for pat, cnt in pcounts.items():
                print(f"    {pat}: {cnt}")

    store.close()


def cmd_doctor(args):
    """Doctor command — self-diagnostic for a nervx brain."""
    import time
    from datetime import datetime, timezone

    repo_root = _resolve_repo(args)
    db = _db_path(repo_root)

    report: dict = {
        "repo_root": repo_root,
        "brain_db": db,
        "checks": [],
    }

    def _check(name: str, status: str, detail: str = "") -> None:
        report["checks"].append({"name": name, "status": status, "detail": detail})

    # 1. brain.db presence
    if not os.path.exists(db):
        _check("brain.db", "FAIL", "not found — run 'nervx build'")
        _render_doctor(args, report)
        return
    _check("brain.db", "OK", db)

    store = GraphStore(db)

    # 2. meta keys
    last_build = store.get_meta("last_build") or ""
    last_update = store.get_meta("last_update") or ""
    file_count = store.get_meta("file_count") or "?"
    node_count = store.get_meta("node_count") or "?"
    edge_count = store.get_meta("edge_count") or "?"
    build_seconds = store.get_meta("build_time_seconds") or "?"
    _check(
        "counts",
        "OK",
        f"{file_count} files, {node_count} nodes, {edge_count} edges",
    )
    _check("build_time_seconds", "OK", build_seconds)

    # 3. brain age
    age_label = "unknown"
    stale_status = "UNKNOWN"
    if last_build:
        try:
            cleaned = last_build.replace("Z", "+00:00")
            built_at = datetime.fromisoformat(cleaned)
            age_s = (datetime.now(timezone.utc) - built_at).total_seconds()
            if age_s < 3600:
                age_label = f"{int(age_s/60)}m ago"
            elif age_s < 86400:
                age_label = f"{age_s/3600:.1f}h ago"
            else:
                age_label = f"{age_s/86400:.1f}d ago"
            stale_status = "OK" if age_s < 7 * 86400 else "WARN"
        except (TypeError, ValueError):
            pass
    _check("last_build", stale_status, f"{last_build or '(never)'} ({age_label})")
    if last_update:
        _check("last_update", "OK", last_update)

    # 4. staleness probe — bounded walk
    stale_files = 0
    sample_size = 0
    try:
        from nervx.build import walk_files
        files = walk_files(repo_root)
        sample_size = min(len(files), 500)
        if last_build:
            try:
                cleaned = last_build.replace("Z", "+00:00")
                ts = datetime.fromisoformat(cleaned).timestamp()
                start = time.monotonic()
                for fp in files[:sample_size]:
                    if time.monotonic() - start > 0.4:
                        break
                    try:
                        if os.path.getmtime(fp) > ts + 1:
                            stale_files += 1
                    except OSError:
                        continue
            except (TypeError, ValueError):
                pass
    except Exception as e:
        _check("staleness_probe", "WARN", str(e))
    else:
        status = "OK" if stale_files == 0 else ("WARN" if stale_files <= 5 else "FAIL")
        _check(
            "staleness",
            status,
            f"{stale_files} file(s) modified since last build (sampled {sample_size})",
        )

    # 5. .nervxignore
    ignore_path = os.path.join(repo_root, ".nervxignore")
    if os.path.exists(ignore_path):
        try:
            nlines = sum(
                1 for ln in open(ignore_path, encoding="utf-8", errors="replace")
                if ln.strip() and not ln.startswith("#")
            )
        except OSError:
            nlines = 0
        _check(".nervxignore", "OK", f"{nlines} patterns")
    else:
        _check(".nervxignore", "INFO", "not present (using built-in defaults)")

    # 6. .git presence
    if os.path.isdir(os.path.join(repo_root, ".git")):
        _check(".git", "OK", "repo has git history (enables diff/cochanges)")
    else:
        _check(".git", "WARN", "no .git directory — cochanges/diff disabled")

    # 7. .nervx gitignore coverage
    gi = os.path.join(repo_root, ".gitignore")
    if os.path.exists(gi):
        try:
            text = open(gi, encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        covered = any(
            p in {l.strip() for l in text.splitlines() if l.strip()}
            for p in (".nervx", ".nervx/", ".nervx/*")
        )
        if covered:
            _check(".gitignore", "OK", ".nervx/ excluded")
        else:
            _check(
                ".gitignore",
                "WARN",
                ".nervx/ not listed — brain.db may leak into commits",
            )
    else:
        _check(".gitignore", "WARN", "no .gitignore found")

    # 8. schema sanity — do the core tables look populated?
    try:
        row = store.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()
        if row and row["c"] > 0:
            _check("schema.nodes", "OK", f"{row['c']} rows")
        else:
            _check("schema.nodes", "FAIL", "0 rows — rebuild required")
    except Exception as e:
        _check("schema.nodes", "FAIL", str(e))

    try:
        row = store.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()
        _check("schema.edges", "OK", f"{row['c']} rows" if row else "0 rows")
    except Exception as e:
        _check("schema.edges", "FAIL", str(e))

    # 9. nervx version
    try:
        from nervx import __version__
        _check("nervx_version", "OK", __version__)
    except Exception:
        pass

    store.close()
    _render_doctor(args, report)


def _render_doctor(args, report: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
        return
    icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗", "INFO": "ℹ", "UNKNOWN": "?"}
    print(f"nervx doctor — {report['repo_root']}")
    for c in report["checks"]:
        sym = icon.get(c["status"], "?")
        line = f"  {sym} {c['name']:<22} {c['status']:<6}"
        if c["detail"]:
            line += f"  {c['detail']}"
        print(line)


def cmd_cochange(args):
    """C19: list co-modified files for a path, optionally with commit ids.

    ``--why`` exposes the actual short commit hashes that link each pair so
    users can `git show <hash>` to inspect the history behind a coupling
    score instead of trusting it blindly.
    """
    repo_root = _resolve_repo(args)
    db = _ensure_brain(repo_root)
    store = GraphStore(db)
    target = args.file.replace("\\", "/")
    rows = store.get_cochanges_for_file(target)

    min_count = max(1, int(getattr(args, "min_count", 3) or 3))
    min_score = float(getattr(args, "min_score", 0.2) or 0.2)
    show_why = bool(getattr(args, "why", False))

    items: list[dict] = []
    for r in rows:
        count = r.get("co_commit_count", 0) or 0
        score = r.get("coupling_score", 0.0) or 0.0
        if count < min_count or score < min_score:
            continue
        other = r["file_b"] if r["file_a"] == target else r["file_a"]
        commit_ids: list[str] = []
        raw = r.get("commit_ids") or "[]"
        try:
            commit_ids = json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            commit_ids = []
        items.append({
            "other": other,
            "co_commit_count": count,
            "coupling_score": round(score, 3),
            "last_co_commit": r.get("last_co_commit") or "",
            "commit_ids": commit_ids,
        })

    if getattr(args, "json", False):
        print(json.dumps({
            "file": target,
            "items": items,
            "min_count": min_count,
            "min_score": min_score,
        }, indent=2, default=str))
        store.close()
        return

    if not items:
        print(f"No co-changes found for {target} "
              f"(min_count={min_count}, min_score={min_score}).")
        store.close()
        return

    print(f"## cochanges: {target}")
    print(f"  {len(items)} coupled file(s)  "
          f"[min_count={min_count}, min_score={min_score}]")
    print("")
    for it in items:
        pct = int(round(it["coupling_score"] * 100))
        print(f"  {it['other']}")
        print(f"    coupling={pct}%  co_commits={it['co_commit_count']}  "
              f"last={it['last_co_commit'] or '?'}")
        if show_why:
            if it["commit_ids"]:
                ids = ", ".join(it["commit_ids"][:10])
                more = ""
                if len(it["commit_ids"]) > 10:
                    more = f" (+{len(it['commit_ids']) - 10} more)"
                print(f"    commits: {ids}{more}")
            else:
                print("    commits: (rebuild brain to capture — older schema)")
        print("")
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
    if getattr(args, "json", False):
        print(json.dumps({
            "output_path": nervx_md,
            "bytes": len(briefing.encode("utf-8")),
        }, indent=2))
    else:
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
    from nervx import __version__

    parser = argparse.ArgumentParser(
        prog="nervx",
        description="A codebase brain for Claude",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
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
    p_nav.add_argument("--verbose-warnings", action="store_true", dest="verbose_warnings",
                       help="Show methodology/confidence/evidence for each warning")
    p_nav.add_argument("--exclude-category", action="append", default=None,
                       dest="exclude_category",
                       help="Drop nodes in this category (vendor, test, generated, example, doc, core). Repeatable.")
    p_nav.add_argument("--include-category", action="append", default=None,
                       dest="include_category",
                       help="Keep only nodes in this category. Repeatable.")
    p_nav.add_argument("--json", action="store_true", help="Emit JSON")
    p_nav.add_argument("--repo", default=None, help="Repository path")
    p_nav.set_defaults(func=cmd_navigate)

    # blast-radius
    p_blast = subparsers.add_parser("blast-radius", aliases=["blast", "br"],
                                     help="Compute blast radius")
    p_blast.add_argument("symbol", help="Symbol ID")
    p_blast.add_argument("--depth", type=int, default=3, help="Max depth")
    p_blast.add_argument("--pick", type=int, default=None,
                         help="Select the Nth candidate from did-you-mean (1-indexed)")
    p_blast.add_argument("--exclude-category", action="append", default=None,
                         dest="exclude_category",
                         help="Drop nodes in this category (vendor, test, generated, example, doc, core). Repeatable.")
    p_blast.add_argument("--include-category", action="append", default=None,
                         dest="include_category",
                         help="Keep only nodes in this category. Repeatable.")
    p_blast.add_argument("--json", action="store_true", help="Emit JSON")
    p_blast.add_argument("--repo", default=None, help="Repository path")
    p_blast.set_defaults(func=cmd_blast_radius)

    # callers
    p_callers = subparsers.add_parser("callers", help="Show what calls a symbol")
    p_callers.add_argument("symbol", help="Symbol ID")
    p_callers.add_argument("--depth", type=int, default=1,
                           help="Caller depth (default: 1)")
    p_callers.add_argument("--pick", type=int, default=None,
                           help="Select the Nth candidate from did-you-mean (1-indexed)")
    p_callers.add_argument("--json", action="store_true", help="Emit JSON")
    p_callers.add_argument("--repo", default=None, help="Repository path")
    p_callers.set_defaults(func=cmd_callers)

    # read
    p_read = subparsers.add_parser("read", help="Read source code of a symbol")
    p_read.add_argument("symbol", help="Symbol ID to read")
    p_read.add_argument("--context", type=int, default=0,
                        help="Include callees up to this depth (default: 0, max: 3)")
    p_read.add_argument("--since", default="",
                        help="Content hash — return 'unchanged' if symbol hasn't changed")
    p_read.add_argument("--pick", type=int, default=None,
                        help="Select the Nth candidate from did-you-mean (1-indexed)")
    p_read.add_argument("--json", action="store_true", help="Emit JSON")
    p_read.add_argument("--repo", default=None, help="Repository path")
    p_read.set_defaults(func=cmd_read)

    # peek
    p_peek = subparsers.add_parser("peek", help="50-token preview of a symbol")
    p_peek.add_argument("symbol", help="Symbol ID")
    p_peek.add_argument("--pick", type=int, default=None,
                        help="Select the Nth candidate from did-you-mean (1-indexed)")
    p_peek.add_argument("--json", action="store_true", help="Emit JSON")
    p_peek.add_argument("--repo", default=None, help="Repository path")
    p_peek.set_defaults(func=cmd_peek)

    # tree
    p_tree = subparsers.add_parser("tree", help="Structural overview of a file")
    p_tree.add_argument("file", help="File path (relative to repo root)")
    p_tree.add_argument("--json", action="store_true", help="Emit JSON")
    p_tree.add_argument("--repo", default=None, help="Repository path")
    p_tree.set_defaults(func=cmd_tree)

    # verify
    p_verify = subparsers.add_parser("verify", help="Check a graph statement")
    p_verify.add_argument("statement", help="e.g. 'foo calls bar'")
    p_verify.add_argument("--json", action="store_true", help="Emit JSON")
    p_verify.add_argument("--repo", default=None, help="Repository path")
    p_verify.set_defaults(func=cmd_verify)

    # ask
    p_ask = subparsers.add_parser("ask", help="Micro-queries about the graph")
    p_ask.add_argument(
        "subcommand",
        choices=sorted([
            "exists", "signature", "calls", "imports", "is-async",
            "returns-type", "callers-count", "has-tests",
        ]),
    )
    p_ask.add_argument("args", nargs="*", help="Subcommand arguments")
    p_ask.add_argument(
        "--pick", type=int, default=None,
        help="Select Nth fuzzy candidate (applies to first symbol arg only for `ask calls`)",
    )
    p_ask.add_argument("--json", action="store_true", help="Emit JSON")
    p_ask.add_argument("--repo", default=None, help="Repository path")
    p_ask.set_defaults(func=cmd_ask)

    # trace
    p_trace = subparsers.add_parser("trace", help="Shortest call path between two symbols")
    p_trace.add_argument("source", help="Source symbol")
    p_trace.add_argument("target", help="Target symbol")
    p_trace.add_argument("--read", action="store_true",
                         help="Include source code of each symbol in the path")
    p_trace.add_argument("--calls-only", action="store_true", dest="calls_only",
                         help="Strict mode: don't fall back to inheritance edges")
    p_trace.add_argument("--via-inheritance", action="store_true", dest="via_inheritance",
                         help="Skip straight to inheritance-aware BFS (soft path)")
    p_trace.add_argument("--pick-source", type=int, default=None, dest="pick_source",
                         help="Select the Nth candidate for the source symbol")
    p_trace.add_argument("--pick-target", type=int, default=None, dest="pick_target",
                         help="Select the Nth candidate for the target symbol")
    p_trace.add_argument("--json", action="store_true", help="Emit JSON")
    p_trace.add_argument("--repo", default=None, help="Repository path")
    p_trace.set_defaults(func=cmd_trace)

    # run
    p_run = subparsers.add_parser("run", help="Run test runners with compact output")
    p_run.add_argument("runner", choices=["pytest"], help="Runner to use")
    p_run.add_argument("pytest_args", nargs=argparse.REMAINDER,
                       help="Arguments to pass to pytest")
    p_run.add_argument("--raw", default="", help="Retrieve raw output by run ID")
    p_run.add_argument("--json", action="store_true", help="Emit JSON wrapper")
    p_run.add_argument("--repo", default=None, help="Repository path")
    p_run.set_defaults(func=cmd_run)

    # string-refs
    p_srefs = subparsers.add_parser(
        "string-refs",
        help="Find quoted string-literal references across languages "
             "(does NOT match bare identifier tokens — for class/function "
             "name references, use `nervx callers` or grep).",
    )
    p_srefs.add_argument(
        "identifier",
        help="Identifier-like literal to search for. Only matches "
             "quoted strings (e.g. \"RadixCache\" in a JSON key or Python "
             "string), not bare code tokens.",
    )
    p_srefs.add_argument("--json", action="store_true", help="Emit JSON")
    p_srefs.add_argument("--repo", default=None, help="Repository path")
    p_srefs.set_defaults(func=cmd_string_refs)

    # flows
    p_flows = subparsers.add_parser("flows", help="Show concept paths")
    p_flows.add_argument("keyword", nargs="?", default=None, help="Filter keyword")
    p_flows.add_argument("--json", action="store_true", help="Emit JSON")
    p_flows.add_argument("--repo", default=None, help="Repository path")
    p_flows.set_defaults(func=cmd_flows)

    # find
    p_find = subparsers.add_parser("find", help="Structural query")
    p_find.add_argument("--kind", choices=["function", "class", "method", "file"])
    p_find.add_argument("--tag", help="Filter by tag")
    p_find.add_argument("--no-tests", action="store_true",
                        help="Only symbols with no test coverage (direct OR 3-hop transitive)")
    p_find.add_argument("--no-direct-tests", action="store_true",
                        help="Only symbols with no DIRECT test calls (strict mode — ignores transitive reach)")
    p_find.add_argument("--importance-gt", type=float, help="Minimum importance")
    p_find.add_argument("--cross-module", action="store_true", help="Only cross-module symbols")
    p_find.add_argument("--dead", action="store_true", help="Only unreferenced symbols (dead code)")
    p_find.add_argument("--exclude-category", action="append", default=None,
                        dest="exclude_category",
                        help="Drop nodes in this category (vendor, test, generated, example, doc, core). Repeatable.")
    p_find.add_argument("--include-category", action="append", default=None,
                        dest="include_category",
                        help="Keep only nodes in this category. Repeatable.")
    p_find.add_argument("--include-test-fixtures", action="store_true",
                        dest="include_test_fixtures",
                        help="Disable the default category auto-filter applied with --no-tests "
                             "(which drops test/doc/example/vendor/generated/script categories)")
    p_find.add_argument("--json", action="store_true", help="Emit JSON")
    p_find.add_argument("--repo", default=None, help="Repository path")
    p_find.set_defaults(func=cmd_find)

    # diff
    p_diff = subparsers.add_parser("diff", help="Show structural changes")
    p_diff.add_argument("--days", type=int, default=7, help="Days to look back")
    p_diff.add_argument("--json", action="store_true", help="Emit JSON")
    p_diff.add_argument("--repo", default=None, help="Repository path")
    p_diff.set_defaults(func=cmd_diff)

    # stats
    p_stats = subparsers.add_parser("stats", help="Print graph statistics")
    p_stats.add_argument("--json", action="store_true", help="Emit JSON")
    p_stats.add_argument("--repo", default=None, help="Repository path")
    p_stats.set_defaults(func=cmd_stats)

    # briefing
    p_brief = subparsers.add_parser("briefing", help="Regenerate NERVX.md")
    p_brief.add_argument("--json", action="store_true", help="Emit JSON")
    p_brief.add_argument("--repo", default=None, help="Repository path")
    p_brief.set_defaults(func=cmd_briefing)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Self-diagnostic for the brain")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON")
    p_doctor.add_argument("--repo", default=None, help="Repository path")
    p_doctor.set_defaults(func=cmd_doctor)

    p_cochange = subparsers.add_parser(
        "cochange", aliases=["cc"],
        help="Co-modified files for a path (add --why to see commit hashes)",
    )
    p_cochange.add_argument("file", help="Source file path (relative to repo root)")
    p_cochange.add_argument("--why", action="store_true",
                            help="Include the short commit hashes behind each coupling")
    p_cochange.add_argument("--min-count", type=int, default=3, dest="min_count",
                            help="Minimum co-commit count (default: 3)")
    p_cochange.add_argument("--min-score", type=float, default=0.2, dest="min_score",
                            help="Minimum coupling score (default: 0.2)")
    p_cochange.add_argument("--json", action="store_true", help="Emit JSON")
    p_cochange.add_argument("--repo", default=None, help="Repository path")
    p_cochange.set_defaults(func=cmd_cochange)

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
