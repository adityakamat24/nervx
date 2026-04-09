"""Build and update orchestration for nervx."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from nervx.memory.store import GraphStore
from nervx.perception.git_miner import GitMiner, is_git_repo
from nervx.perception.ignore import load_ignore_patterns, should_ignore
from nervx.perception.languages import get_supported_extensions
from nervx.perception.linker import collect_raw_imports, resolve_all
from nervx.perception.parser import extract_keywords, parse_file

# Directories to exclude during file walk.
#
# The "noise" bucket (benchmark, third_party, examples, ...) is excluded by
# default because their code is rarely what a user cares about when querying
# the brain — and critically, they flood the importance-score distribution
# with high-fanout dunder/vendored functions that crowd out real load-bearing
# code. Users who want them indexed can add a negation to `.nervxignore`
# (e.g. `!benchmark/`).
EXCLUDE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".eggs", "dist", "build",
    ".nervx", ".egg-info",
    # Java/Gradle/Maven
    ".gradle", ".mvn", "target", "out",
    # C#/.NET
    "bin", "obj", "Debug", "Release", "packages",
    # Go
    "vendor", "pkg",
    # Rust
    # ("target" already included above)
    # Ruby
    ".bundle",
    # General noise
    ".idea", ".vscode", ".vs", "coverage", ".nyc_output",
    # Noise bucket — rarely what users query for, floods importance distribution
    "third_party", "third-party", "3rdparty", "thirdparty",
    "benchmark", "benchmarks", "bench",
    "examples", "example", "samples", "sample",
    "fixtures", "testdata", "test_data",
})

# Files to exclude
EXCLUDE_FILES = frozenset({
    "setup.py", "setup.cfg", "conftest.py",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.sum", "Cargo.lock", "Gemfile.lock",
})

# Max file size (500KB)
MAX_FILE_SIZE = 500 * 1024


# All supported source file extensions — driven by the language registry
# so adding a language only requires updating nervx/perception/languages.py.
ALL_EXTENSIONS = tuple(sorted(get_supported_extensions()))


# C17: path-based categories. Exactly one category tag is assigned per node
# based on its file path, and it's stored as ``category:<name>`` in the tags
# list so existing tag-filter plumbing picks it up for free.
#
# Order matters: the first matching prefix or pattern wins. "core" is the
# fallback when nothing else matches.
_CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    # category, substrings that must appear as a path segment (or suffix)
    ("vendor", ("/node_modules/", "/vendor/", "/third_party/",
                "/3rdparty/", "/thirdparty/", "/third-party/")),
    ("generated", ("/dist/", "/build/", "/.next/", "/target/", "/out/",
                   ".min.js", ".min.css", ".bundle.js", ".generated.",
                   "_pb.py", "_pb2.py", ".pb.go")),
    ("test", ("/tests/", "/test/", "/__tests__/", "/spec/", "/specs/",
              "_test.", ".test.", ".spec.", "test_", "_spec.")),
    # CI / build scripts — rarely load-bearing, flood importance scores.
    # Deliberately does NOT match "/tools/" (nervx's own runners live under
    # nervx/tools/) or "/utils/" (too generic).
    ("script", ("/scripts/", "/ci/", "/build_scripts/", "/.github/")),
    ("example", ("/examples/", "/example/", "/samples/", "/sample/",
                 "/demos/", "/demo/")),
    ("doc", ("/docs/", "/doc/", "/documentation/")),
]


def _derive_category(file_path: str) -> str:
    """Return a single category name for a source file.

    ``file_path`` is expected in forward-slash form. Missing slashes are
    fine — we also check suffixes so patterns like ``_test.py`` still hit.
    """
    fp = file_path.replace("\\", "/").lower()
    # Normalize so leading-segment patterns ("/tests/") work even when the
    # path is relative and doesn't start with a slash.
    normalized = "/" + fp if not fp.startswith("/") else fp
    for category, patterns in _CATEGORY_PATTERNS:
        for p in patterns:
            if p.startswith("/") and p in normalized:
                return category
            if not p.startswith("/") and (p in fp or fp.endswith(p)):
                # Suffix/substring pattern — check last path segment for
                # `test_` / `_test.` prefixes so `controller_test.go` hits
                # but `fast_testing_util.go` doesn't.
                if p.startswith("_") or p.startswith(".") or p.endswith("."):
                    return category
                # Bare-prefix patterns like "test_" must be at the start of
                # the basename, not buried mid-name.
                base = fp.rsplit("/", 1)[-1]
                if base.startswith(p):
                    return category
    return "core"


def walk_files(repo_root: str, extensions: tuple[str, ...] = ALL_EXTENSIONS) -> list[str]:
    """Walk the repo and collect source files, applying exclusion rules.

    Exclusion is driven by:
      1. Hardcoded fast-path EXCLUDE_DIRS / EXCLUDE_FILES (prune walk for speed).
      2. `.nervxignore` + built-in defaults via `should_ignore` (gitignore semantics).
    """
    ignore_patterns = load_ignore_patterns(repo_root)
    # If there are any negation patterns, we cannot prune directories — gitignore
    # negations need to see children of an excluded directory to re-include them.
    has_negations = any(p.startswith("!") for p in ignore_patterns)
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Fast-prune obvious noise directories from the walk.
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
        ]

        # Apply .nervxignore to directory pruning too — avoids descending into
        # user-ignored trees like `generated/` or `vendor/`. Skipped if there
        # are negation patterns, since negations may re-include children.
        if not has_negations:
            dirnames[:] = [
                d for d in dirnames
                if not should_ignore(
                    _rel_path(os.path.join(dirpath, d), repo_root) + "/",
                    ignore_patterns,
                )
            ]

        for f in filenames:
            if f in EXCLUDE_FILES:
                continue
            if not any(f.endswith(ext) for ext in extensions):
                continue
            full_path = os.path.join(dirpath, f)
            rel = _rel_path(full_path, repo_root)
            if should_ignore(rel, ignore_patterns):
                continue
            try:
                if os.path.getsize(full_path) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            files.append(full_path)

    return files


def _file_hash(path: str) -> str:
    """Compute MD5 hash of file contents."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _rel_path(file_path: str, repo_root: str) -> str:
    """Compute forward-slash relative path."""
    try:
        rel = os.path.relpath(file_path, repo_root)
    except ValueError:
        rel = file_path
    return rel.replace("\\", "/")


# Identifier-like literal pattern — matches `foo_bar`, `voteTarget`, `user_id`.
# Kept intentionally narrow to avoid indexing prose or short noise.
_IDENT_LITERAL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,63}$")
# Quoted string in a line; captures the *content* of the quotes.
_QUOTED_STRING_RE = re.compile(r"""["'`]([^"'`\n\\]{3,64})["'`]""")


def compute_symbol_hashes_and_strings(
    store: GraphStore, source_files: list[str], repo_root: str
) -> None:
    """Two jobs in one file-read pass per file:

    1. Hash each symbol's source text (md5, 8 hex chars) → nodes.content_hash.
    2. Index identifier-like string literals → string_refs table.
    """
    # Group nodes by file so we only read each file once.
    all_nodes = store.get_all_nodes()
    by_file: dict[str, list[dict]] = {}
    for n in all_nodes:
        if n["kind"] == "file":
            continue
        by_file.setdefault(n["file_path"], []).append(n)

    hash_updates: list[tuple[str, str]] = []
    string_rows: list[tuple[str, str, int, str]] = []

    for fp in source_files:
        rel = _rel_path(fp, repo_root)
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()

        # Phase 1: symbol content hashes
        for node in by_file.get(rel, []):
            ls = node.get("line_start") or 0
            le = node.get("line_end") or 0
            if not ls or not le:
                continue
            start_idx = max(0, ls - 1)
            end_idx = min(len(lines), le)
            if start_idx >= end_idx:
                continue
            source = "\n".join(lines[start_idx:end_idx])
            digest = hashlib.md5(source.encode("utf-8", errors="replace")).hexdigest()[:8]
            hash_updates.append((digest, node["id"]))

        # Phase 2: string-literal cross-reference — language-agnostic heuristic:
        # pull quoted strings, keep those that look like identifiers.
        for line_num, line in enumerate(lines, start=1):
            # Skip lines that are clearly comments (conservative — language
            # detection would be more precise, but this handles the common case).
            stripped = line.lstrip()
            if stripped.startswith(("#", "//", "*", "/*")):
                continue
            for m in _QUOTED_STRING_RE.finditer(line):
                literal = m.group(1)
                if _IDENT_LITERAL_RE.match(literal):
                    string_rows.append((literal, rel, line_num, "string_literal"))

    if hash_updates:
        store.conn.executemany(
            "UPDATE nodes SET content_hash = ? WHERE id = ?",
            hash_updates,
        )
    if string_rows:
        # Dedup rows (same literal/file/line may match multiple times on one line)
        unique = list({(r[0], r[1], r[2]): r for r in string_rows}.values())
        store.add_string_refs_bulk(unique)
    store.conn.commit()


def compute_importance(store: GraphStore):
    """Compute importance scores and percentile ranks for all nodes.

    Formula: `calls_in*2 + inherits_in*1.5 + imports_in*0.5 + cross_module*2
              + is_exported + calls_out*0.5`.
    Reverse edges (`called_by`/`inherited_by`/`imported_by`) are skipped to
    avoid double-counting (they're just the inverse of forward edges).
    Importance is a rough "should a human care about this symbol" signal,
    not a correctness metric — use `importance_rank` (0-100 percentile) in
    output to avoid giving users the impression that raw scores are
    comparable across codebases of different sizes.
    """
    nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()

    # Per-edge-type incoming weights. Reverse edges skipped so a node's
    # incoming call count isn't also counted as its outgoing called_by count.
    FORWARD_WEIGHT = {
        "calls": 2.0,
        "inherits": 1.5,
        "imports": 0.5,
        "instantiates": 1.0,
    }

    # Build per-type in-degree and a general out-degree for calls only.
    in_weighted: dict[str, float] = {}
    out_calls: dict[str, int] = {}
    for e in all_edges:
        etype = e["edge_type"]
        w = FORWARD_WEIGHT.get(etype)
        if w is None:
            continue  # skip called_by/imported_by/inherited_by — reverse edges
        in_weighted[e["target_id"]] = in_weighted.get(e["target_id"], 0.0) + w
        if etype == "calls":
            out_calls[e["source_id"]] = out_calls.get(e["source_id"], 0) + 1

    # Build node -> module map
    node_module: dict[str, str] = {}
    for n in nodes:
        fp = n["file_path"]
        node_module[n["id"]] = fp.split("/")[0] if "/" in fp else ""

    # Count cross-module forward edges per node (forward only — same reason).
    cross_module: dict[str, int] = {}
    for e in all_edges:
        if e["edge_type"] not in FORWARD_WEIGHT:
            continue
        src_mod = node_module.get(e["source_id"], "")
        tgt_mod = node_module.get(e["target_id"], "")
        if src_mod and tgt_mod and src_mod != tgt_mod:
            cross_module[e["source_id"]] = cross_module.get(e["source_id"], 0) + 1
            cross_module[e["target_id"]] = cross_module.get(e["target_id"], 0) + 1

    # Raw importance per node.
    raw: list[tuple[str, float]] = []
    for node in nodes:
        nid = node["id"]
        is_exported = 1.0 if not node["name"].startswith("_") else 0.0
        score = (
            in_weighted.get(nid, 0.0)
            + cross_module.get(nid, 0) * 2.0
            + is_exported
            + out_calls.get(nid, 0) * 0.5
        )
        raw.append((nid, score))

    # Percentile rank: 0-100 based on ordinal position. Ties share the lower rank.
    sorted_scores = sorted(raw, key=lambda x: x[1])
    n_total = len(sorted_scores)
    rank_map: dict[str, int] = {}
    if n_total > 0:
        for i, (nid, _) in enumerate(sorted_scores):
            # i=0 → lowest rank 0, i=n-1 → rank 100
            rank_map[nid] = int(round((i / max(1, n_total - 1)) * 100))

    updates = [(score, rank_map.get(nid, 0), nid) for nid, score in raw]
    store.conn.executemany(
        "UPDATE nodes SET importance = ?, importance_rank = ? WHERE id = ?",
        updates,
    )
    store.conn.commit()


def full_build(repo_root: str, db_path: str):
    """Run a full build of the nervx brain.

    Phases:
    1. Walk repo, find source files
    2. Parse each file with tree-sitter
    3. Store nodes and keywords
    4. Resolve edges (link)
    5. Mine git history
    6. Compute importance scores
    7. Store metadata
    """
    start = time.monotonic()
    store = GraphStore(db_path)
    store.clear_all()

    # Phase 1: Walk
    source_files = walk_files(repo_root)
    if not source_files:
        store.set_meta("repo_root", repo_root)
        store.set_meta("file_count", "0")
        store.set_meta("node_count", "0")
        store.set_meta("edge_count", "0")
        store.set_meta("last_build", datetime.now(timezone.utc).isoformat())
        store.set_meta("build_time_seconds", str(round(time.monotonic() - start, 2)))
        store.close()
        return store

    # Phase 2: Parse
    parse_results = []
    now_iso = datetime.now(timezone.utc).isoformat()
    n_files = len(source_files)
    for i, fp in enumerate(source_files):
        if n_files > 100 and (i + 1) % 200 == 0:
            print(f"  Parsing... {i + 1}/{n_files}", file=sys.stderr)
        try:
            pr = parse_file(fp, repo_root)
            parse_results.append(pr)
        except Exception as e:
            print(f"Warning: failed to parse {fp}: {e}", file=sys.stderr)

    # Phase 3: Store nodes and keywords, record file hashes (batched)
    with store.batch():
        for pr in parse_results:
            for node in pr.nodes:
                tags = list(node.tags)
                category = _derive_category(node.file_path)
                cat_tag = f"category:{category}"
                if cat_tag not in tags:
                    tags.append(cat_tag)
                store.upsert_node(
                    id=node.id, kind=node.kind, name=node.name,
                    file_path=node.file_path, line_start=node.line_start,
                    line_end=node.line_end, signature=node.signature,
                    docstring=node.docstring, tags=tags,
                    parent_id=node.parent_id,
                )
                kws = extract_keywords(node)
                if kws:
                    store.add_keywords_bulk([(kw, node.id, src) for kw, src in kws])

        # Record file hashes
        for fp in source_files:
            rel = _rel_path(fp, repo_root)
            store.upsert_file_hash(rel, _file_hash(fp), now_iso)

    # Phase 3b: Content hashes for --since diffing, and string-literal index.
    try:
        compute_symbol_hashes_and_strings(store, source_files, repo_root)
    except Exception as e:
        print(f"Warning: content hashing/string index failed: {e}", file=sys.stderr)

    # Phase 4: Resolve edges
    edges = resolve_all(parse_results)
    with store.batch():
        for e in edges:
            store.add_edge(e.source_id, e.target_id, e.edge_type, e.weight, e.metadata)

    # Phase 4b (0.2.6): persist every raw import — intra-project AND
    # external — so `ask imports <file>` can show numpy/torch/std imports
    # that never produced an intra-project edge.
    try:
        raw_import_rows = collect_raw_imports(parse_results)
        if raw_import_rows:
            store.add_raw_imports_bulk(raw_import_rows)
    except Exception as e:
        print(f"Warning: raw-imports persistence failed: {e}", file=sys.stderr)

    # Phase 5: Mine git
    if is_git_repo(repo_root):
        miner = GitMiner(repo_root)
        try:
            with store.batch():
                miner.mine(store)
        except Exception as e:
            print(f"Warning: git mining failed: {e}", file=sys.stderr)
    else:
        print("Warning: not a git repository, skipping git mining.", file=sys.stderr)

    # Phase 6: Compute importance
    compute_importance(store)

    # Phase 7: Detect patterns
    try:
        from nervx.instinct.patterns import detect_patterns
        detect_patterns(store)
    except Exception as e:
        print(f"Warning: pattern detection failed: {e}", file=sys.stderr)

    # Phase 8: Analyze contracts
    try:
        from nervx.reflexes.warnings import analyze_contracts
        with store.batch():
            analyze_contracts(store, parse_results)
    except Exception as e:
        print(f"Warning: contract analysis failed: {e}", file=sys.stderr)

    # Phase 9: Detect concept paths
    try:
        from nervx.attention.concepts import detect_concept_paths
        detect_concept_paths(store)
    except Exception as e:
        print(f"Warning: concept path detection failed: {e}", file=sys.stderr)

    # Phase 10: Store metadata
    node_count = len(store.get_all_nodes())
    edge_count = len(store.conn.execute("SELECT * FROM edges").fetchall())

    store.set_meta("repo_root", repo_root)
    store.set_meta("file_count", str(len(source_files)))
    store.set_meta("node_count", str(node_count))
    store.set_meta("edge_count", str(edge_count))
    store.set_meta("last_build", now_iso)
    store.set_meta("build_time_seconds", str(round(time.monotonic() - start, 2)))

    elapsed = round(time.monotonic() - start, 2)
    print(f"Built nervx: {len(source_files)} files, {node_count} symbols, "
          f"{edge_count} edges in {elapsed}s")

    store.close()
    return db_path


def incremental_update(repo_root: str, db_path: str):
    """Incrementally update the brain by re-parsing changed files."""
    start = time.monotonic()
    store = GraphStore(db_path)

    source_files = walk_files(repo_root)
    old_hashes = store.get_all_file_hashes()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Detect changes
    current_files = {}
    for fp in source_files:
        rel = _rel_path(fp, repo_root)
        current_files[rel] = _file_hash(fp)

    changed = []
    new = []
    for rel, h in current_files.items():
        if rel not in old_hashes:
            new.append(rel)
        elif old_hashes[rel] != h:
            changed.append(rel)

    deleted = [rel for rel in old_hashes if rel not in current_files]

    if not changed and not new and not deleted:
        print("No changes detected.")
        store.close()
        return

    # Clear data for changed/deleted files
    for rel in changed + deleted:
        store.clear_file_data(rel)

    # Re-parse ALL files (needed for correct edge resolution)
    parse_results = []
    for fp in source_files:
        try:
            pr = parse_file(fp, repo_root)
            parse_results.append(pr)
        except Exception as e:
            print(f"Warning: failed to parse {fp}: {e}", file=sys.stderr)

    # Clear all edges (we re-resolve from scratch)
    store.conn.execute("DELETE FROM edges")
    store.conn.commit()

    # Store nodes and keywords for changed/new files (batched)
    changed_set = set(changed + new)
    with store.batch():
        for pr in parse_results:
            if pr.file_path in changed_set:
                for node in pr.nodes:
                    tags = list(node.tags)
                    cat_tag = f"category:{_derive_category(node.file_path)}"
                    if cat_tag not in tags:
                        tags.append(cat_tag)
                    store.upsert_node(
                        id=node.id, kind=node.kind, name=node.name,
                        file_path=node.file_path, line_start=node.line_start,
                        line_end=node.line_end, signature=node.signature,
                        docstring=node.docstring, tags=tags,
                        parent_id=node.parent_id,
                    )
                    kws = extract_keywords(node)
                    if kws:
                        store.add_keywords_bulk([(kw, node.id, src) for kw, src in kws])

        # Update file hashes
        for rel, h in current_files.items():
            store.upsert_file_hash(rel, h, now_iso)
        for rel in deleted:
            store.conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (rel,))

    # Re-resolve all edges
    edges = resolve_all(parse_results)
    with store.batch():
        for e in edges:
            store.add_edge(e.source_id, e.target_id, e.edge_type, e.weight, e.metadata)

    # 0.2.6: refresh raw_imports table too — clear all rows then re-bulk
    # insert. Simpler than incremental per-file diffing and cheap compared
    # to a full re-parse.
    try:
        store.conn.execute("DELETE FROM raw_imports")
        store.conn.commit()
        raw_import_rows = collect_raw_imports(parse_results)
        if raw_import_rows:
            store.add_raw_imports_bulk(raw_import_rows)
    except Exception as e:
        print(f"Warning: raw-imports persistence failed: {e}", file=sys.stderr)

    # Re-compute content hashes and string-literal index for all files.
    # Cheap enough (one file read per source file) and guarantees --since
    # and string-refs stay accurate after incremental edits.
    try:
        compute_symbol_hashes_and_strings(store, source_files, repo_root)
    except Exception as e:
        print(f"Warning: content hashing/string index failed: {e}", file=sys.stderr)

    # Re-mine git
    if is_git_repo(repo_root):
        store.conn.execute("DELETE FROM file_stats")
        store.conn.execute("DELETE FROM cochanges")
        store.conn.commit()
        miner = GitMiner(repo_root)
        try:
            with store.batch():
                miner.mine(store)
        except Exception:
            pass

    # Recompute importance
    compute_importance(store)

    # Update metadata
    node_count = len(store.get_all_nodes())
    edge_count = len(store.conn.execute("SELECT * FROM edges").fetchall())
    store.set_meta("node_count", str(node_count))
    store.set_meta("edge_count", str(edge_count))
    store.set_meta("last_update", now_iso)

    elapsed = round(time.monotonic() - start, 2)
    print(f"Updated nervx: {len(changed)} changed, {len(new)} new, "
          f"{len(deleted)} deleted in {elapsed}s")

    store.close()
