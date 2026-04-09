"""Query engine: navigate, find, blast-radius, diff."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from nervx.memory.store import GraphStore
from nervx.reflexes.warnings import Warning, collect_warnings, compute_blast_radius


# ── Synonym expansion ─────────────────────────────────────────────

_SYNONYMS_CACHE: dict[str, list[str]] | None = None


def _load_synonyms() -> dict[str, list[str]]:
    """Load static synonym map once and cache it."""
    global _SYNONYMS_CACHE
    if _SYNONYMS_CACHE is not None:
        return _SYNONYMS_CACHE
    path = Path(__file__).parent / "synonyms.json"
    try:
        _SYNONYMS_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _SYNONYMS_CACHE = {}
    return _SYNONYMS_CACHE


def _expand_synonyms(terms: list[str]) -> list[str]:
    """Return terms plus all synonyms, preserving order and deduping."""
    syns = _load_synonyms()
    expanded: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t not in seen:
            expanded.append(t)
            seen.add(t)
        for s in syns.get(t, []):
            if s not in seen:
                expanded.append(s)
                seen.add(s)
    return expanded


# ── Query stop words ──────────────────────────────────────────────

QUERY_STOP_WORDS = frozenset({
    # Natural language noise — NOT code-relevant terms
    "the", "a", "an", "to", "of", "in", "for", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "how", "what", "where", "when", "why", "which", "who",
    "can", "could", "should", "would", "need", "want", "like",
    "i", "my", "me", "we", "our", "it", "its", "this", "that",
    "and", "or", "but", "not", "with", "from", "into",
    "help", "please", "just", "some", "any", "about", "look", "tell",
    "show", "find", "where", "there", "here", "very", "also",
})

_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z0-9])([A-Z])")


@dataclass
class NavigateResult:
    terms: list[str]
    primary: list[dict]
    secondary: list[dict]
    cochange_files: list[dict]
    read_order: list[dict]
    warnings: list[Warning]
    flows: list[dict] = field(default_factory=list)
    formatted: str = ""
    # Map of secondary node_id -> relationship (edge_type or "docstring-match")
    secondary_edge: dict[str, str] = field(default_factory=dict)


def _tokenize_query(query: str) -> list[str]:
    """Split query into tokens, remove stop words."""
    # Split on whitespace
    raw_tokens = query.split()
    tokens = []

    for token in raw_tokens:
        # Split camelCase
        s = _CAMEL_RE1.sub(r"\1_\2", token)
        s = _CAMEL_RE2.sub(r"\1_\2", s)
        # Split snake_case
        parts = s.split("_")
        for p in parts:
            w = p.lower().strip()
            if w and w not in QUERY_STOP_WORDS and len(w) >= 2:
                tokens.append(w)

    return list(dict.fromkeys(tokens))  # dedupe preserving order


_COMMON_SUFFIXES = re.compile(r"(ing|tion|ment|ness|able|ible|ous|ive|ed|er|es|ly|al|ful)$")
_TRAILING_S = re.compile(r"(?<![s])s$")  # strip trailing "s" but not "ss"


def _stem(word: str) -> str:
    """Minimal stemmer: strip common English suffixes for prefix matching."""
    s = _COMMON_SUFFIXES.sub("", word)
    if s == word:
        # Try stripping plural "s" (but not "ss" like "class")
        s = _TRAILING_S.sub("", word)
    # Don't over-stem — keep at least 3 chars
    return s if len(s) >= 3 else word


def _node_category(tags_raw) -> str:
    """Extract ``category:<name>`` tag or return empty string."""
    try:
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
    except (TypeError, ValueError):
        return ""
    for t in tags:
        if isinstance(t, str) and t.startswith("category:"):
            return t.split(":", 1)[1].lower()
    return ""


def navigate(
    store: GraphStore,
    query: str,
    budget: int = 5,
    exclude_category: list[str] | None = None,
    include_category: list[str] | None = None,
) -> NavigateResult:
    """Core navigate query: given a natural language task, return ranked symbols."""

    # Step 1: Tokenize
    terms = _tokenize_query(query)
    if not terms:
        return NavigateResult(
            terms=[], primary=[], secondary=[], cochange_files=[],
            read_order=[], warnings=[], formatted="No searchable terms found.",
        )

    # Step 2: Seed finding — exact + prefix/stem search, merged
    exact_results = store.search_keywords_weighted(terms)
    exact_ids = {nid for nid, _ in exact_results}

    # Always also try prefix/stem matching to catch morphological variants
    # ("disconnects" → "disconnect", "handling" → "handl" → "handle")
    stems = list(dict.fromkeys(_stem(t) for t in terms))
    prefix_results = store.search_keywords_prefix(stems)

    # Synonym expansion — only lookups for terms the user didn't already type.
    expanded = _expand_synonyms(terms)
    synonym_only = [t for t in expanded if t not in terms]
    synonym_results = (
        store.search_keywords_weighted(synonym_only) if synonym_only else []
    )

    # Merge: exact matches keep full score, prefix-only matches get reduced weight
    seed_map: dict[str, float] = {}
    for nid, score in exact_results:
        seed_map[nid] = score
    for nid, score in prefix_results:
        if nid in seed_map:
            # Boost exact matches that also have prefix hits
            seed_map[nid] = max(seed_map[nid], score)
        else:
            seed_map[nid] = score * 0.6  # prefix-only gets 60% weight
    # Synonym hits get half weight — they're suggestive, not authoritative.
    for nid, score in synonym_results:
        boosted = score * 0.5
        if nid in seed_map:
            seed_map[nid] = max(seed_map[nid], boosted)
        else:
            seed_map[nid] = boosted

    seed_results = sorted(seed_map.items(), key=lambda x: -x[1])

    # Final fallback: direct node name search (if still too few)
    if len(seed_results) < budget:
        name_results = store.search_nodes_by_name(terms)
        existing_ids = {nid for nid, _ in seed_results}
        for nid, match_count in name_results:
            if nid not in existing_ids:
                seed_results.append((nid, match_count * 2.0))
                existing_ids.add(nid)

    exclude_set = {c.strip().lower() for c in (exclude_category or []) if c.strip()}
    include_set = {c.strip().lower() for c in (include_category or []) if c.strip()}

    # Enrich with node data and importance, applying category filters here
    # so docstring-demotion and cohesion-penalty only see the allowed set.
    scored_seeds = []
    for node_id, match_score in seed_results[:40]:  # wider window pre-filter
        node = store.get_node(node_id)
        if node is None:
            continue
        if exclude_set or include_set:
            cat = _node_category(node.get("tags"))
            if exclude_set and cat in exclude_set:
                continue
            if include_set and cat not in include_set:
                continue
        scored_seeds.append((node, match_score))
        if len(scored_seeds) >= 20:
            break

    # Step 3: Score and rank.
    # Keyword relevance dominates; importance is a weak tiebreaker; and a
    # term-coverage bonus rewards candidates that match MORE of the user's
    # query terms. The ranking is scale-invariant — no hardcoded importance
    # cutoffs — so it behaves the same on a tiny script or a monorepo.
    term_set_lc = {t.lower() for t in terms}

    def _matched_terms(node) -> int:
        name_lc = (node["name"] or "").lower()
        path_lc = (node["file_path"] or "").lower()
        # Tokenize name + path using the same splitter the query uses so
        # camelCase and snake_case both count.
        tokens = set(_tokenize_query(f"{name_lc} {path_lc}"))
        return len({t for t in term_set_lc if t in tokens or t in name_lc or t in path_lc})

    def _score(node, match_score):
        kind_bonus = 1.5 if node["kind"] in ("function", "method") else 0.5
        coverage = _matched_terms(node)
        return (
            match_score * 2.0
            + node["importance"] * 0.15
            + kind_bonus
            + coverage * 1.8
        )

    # Pre-compute coverage per node so we can apply a hard partition after
    # both the _score sort AND the cohesion penalty (step 3b) have run:
    # whenever at least one candidate matches 2+ query terms, single-term
    # candidates can't jump above multi-term ones. This is the real fix
    # for concept-name collisions — e.g. a query like "X form Y from Z"
    # should prefer a node matching two of {X, Y, Z} over a node matching
    # only X, regardless of how important X-only is.
    coverage_map: dict[str, int] = {
        node["id"]: _matched_terms(node) for node, _ in scored_seeds
    }
    max_coverage = max(coverage_map.values(), default=0)

    scored_seeds.sort(key=lambda x: -_score(x[0], x[1]))

    # Partition into strong-signal and weak-signal hits. A strong hit has at
    # least one query term in the node name OR the file path — a docstring-only
    # match isn't enough to stay in the primary list (it becomes secondary).
    strong: list[tuple[dict, float]] = []
    weak_demoted: list[dict] = []
    for node, ms in scored_seeds:
        name_lc = (node["name"] or "").lower()
        path_lc = (node["file_path"] or "").lower()
        if any(t in name_lc or t in path_lc for t in term_set_lc):
            strong.append((node, ms))
        else:
            weak_demoted.append(node)

    # Step 3b: Cross-package cohesion — if the top strong results cluster in
    # a common path prefix, penalize candidates whose file path shares no
    # tokens with that prefix. This kills cases like `nav "radix insert"`
    # matching a filesystem `insert` helper in an unrelated package.
    def _path_tokens(path: str) -> set[str]:
        return {t for t in re.split(r"[\\/._-]", path.lower()) if t and t not in {"py", "js", "ts", "go", "rs", "java", "c", "cpp", "h"}}

    top_strong = strong[:budget]
    if len(top_strong) >= 2:
        prefix_counts: dict[str, int] = defaultdict(int)
        for n, _ in top_strong:
            parts = (n["file_path"] or "").replace("\\", "/").split("/")
            # Count each directory ancestor once
            for depth in range(1, min(len(parts), 4)):
                prefix_counts["/".join(parts[:depth])] += 1
        shared_prefix = ""
        for pref, cnt in prefix_counts.items():
            if cnt >= max(2, len(top_strong) // 2 + 1) and len(pref) > len(shared_prefix):
                shared_prefix = pref
        if shared_prefix:
            shared_tokens = _path_tokens(shared_prefix)
            if shared_tokens:
                penalized: list[tuple[dict, float]] = []
                for node, ms in strong:
                    node_tokens = _path_tokens(node["file_path"] or "")
                    if shared_tokens & node_tokens:
                        penalized.append((node, ms))
                    else:
                        # No overlap with the cohesion cluster — big penalty so
                        # it rarely survives the top-N cut but is still visible
                        # if nothing better exists.
                        penalized.append((node, ms * 0.3))
                penalized.sort(key=lambda x: -_score(x[0], x[1]))
                strong = penalized

    # Step 3c: Coverage-tier partition. Applied AFTER cohesion so the
    # re-sort on raw _score can't put a lower-coverage high-importance
    # distractor back at the top. Scale-invariant and generalized across
    # any number of query terms: candidates matching the maximum number
    # of query terms always beat candidates matching fewer, regardless of
    # importance gap. Within a tier, _score (importance + kind + cohesion)
    # still decides order. Lower-coverage candidates stay visible in the
    # tail and the secondary list.
    if max_coverage >= 2:
        best = [
            (n, ms) for n, ms in strong
            if coverage_map.get(n["id"], 0) == max_coverage
        ]
        rest = [
            (n, ms) for n, ms in strong
            if coverage_map.get(n["id"], 0) < max_coverage
        ]
        strong = best + rest

    # Step 4: Take top N strong hits as primary
    primary = [s[0] for s in strong[:budget]]
    primary_ids = {n["id"] for n in primary}

    # Step 5: Expand neighborhood. Keep (node, edge_type) tuples so
    # _format_navigate can label the relationship in the "Connected Symbols"
    # section (B6) — a raw node list hides *why* it's in the result.
    secondary: list[dict] = []
    secondary_ids: set[str] = set()
    secondary_edge: dict[str, str] = {}
    for p in primary:
        edges = store.get_edges_from(p["id"]) + store.get_edges_to(p["id"])
        for e in edges:
            other_id = e["target_id"] if e["source_id"] == p["id"] else e["source_id"]
            if other_id not in primary_ids and other_id not in secondary_ids:
                other = store.get_node(other_id)
                if other and other["kind"] != "file":
                    secondary.append(other)
                    secondary_ids.add(other_id)
                    secondary_edge[other_id] = e["edge_type"]
                    if len(secondary) >= 8:
                        break
        if len(secondary) >= 8:
            break

    # Append any weak/docstring-only matches that didn't make the primary cut
    # so they remain discoverable. Tag them with "docstring-match".
    for node in weak_demoted:
        if len(secondary) >= 12:
            break
        if node["id"] in primary_ids or node["id"] in secondary_ids:
            continue
        if node["kind"] == "file":
            continue
        secondary.append(node)
        secondary_ids.add(node["id"])
        secondary_edge[node["id"]] = "docstring-match"

    # Step 6: Co-change files. B5 — always include co_commit_count so the
    # formatter can suppress statistically weak relationships (n < 3) instead
    # of reporting misleading "50% coupling" for two coincidental commits.
    cochange_files = []
    seen_files = {n["file_path"] for n in primary + secondary}
    for p in primary:
        cochanges = store.get_cochanges_for_file(p["file_path"])
        for cc in cochanges:
            if cc["coupling_score"] >= 0.4 and cc.get("co_commit_count", 0) >= 3:
                other = cc["file_b"] if cc["file_a"] == p["file_path"] else cc["file_a"]
                if other not in seen_files:
                    cochange_files.append({
                        "file_path": other,
                        "coupling_score": cc["coupling_score"],
                        "co_commit_count": cc.get("co_commit_count", 0),
                    })
                    seen_files.add(other)
                    if len(cochange_files) >= 3:
                        break
        if len(cochange_files) >= 3:
            break

    # Step 7: Warnings
    warnings = collect_warnings(store, [n["id"] for n in primary])

    # Step 8: Compute read order (topological sort of primary by calls edges)
    read_order = _compute_read_order(store, primary)

    # Step 9: Trace execution flows from primary nodes
    flows = _trace_flows(store, primary)

    # Step 10: Format
    formatted = _format_navigate(terms, primary, secondary, cochange_files,
                                  read_order, warnings, flows, secondary_edge)

    return NavigateResult(
        terms=terms,
        primary=primary,
        secondary=secondary,
        cochange_files=cochange_files,
        read_order=read_order,
        warnings=warnings,
        flows=flows,
        formatted=formatted,
        secondary_edge=secondary_edge,
    )


def _compute_read_order(store: GraphStore, nodes: list[dict]) -> list[dict]:
    """Topological sort of nodes by calls edges. Break cycles by importance."""
    if not nodes:
        return []

    node_map = {n["id"]: n for n in nodes}
    ids = set(node_map.keys())

    # Build adjacency: A calls B within the primary set
    adj: dict[str, set[str]] = defaultdict(set)
    in_deg: dict[str, int] = {nid: 0 for nid in ids}

    for nid in ids:
        edges = store.get_edges_from(nid)
        for e in edges:
            if e["edge_type"] == "calls" and e["target_id"] in ids:
                adj[nid].add(e["target_id"])
                in_deg[e["target_id"]] += 1

    # Kahn's algorithm with importance-based tie breaking
    queue = sorted(
        [nid for nid, d in in_deg.items() if d == 0],
        key=lambda x: -node_map[x]["importance"],
    )
    result = []

    while queue:
        nid = queue.pop(0)
        result.append(node_map[nid])
        for neighbor in adj[nid]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
                queue.sort(key=lambda x: -node_map[x]["importance"])

    # Add any remaining nodes (cycles) sorted by importance
    remaining = [node_map[nid] for nid in ids if nid not in {n["id"] for n in result}]
    remaining.sort(key=lambda n: -n["importance"])
    result.extend(remaining)

    return result


def _trace_flows(
    store: GraphStore,
    primary: list[dict],
    max_depth: int = 4,
    max_paths: int = 1,
    max_branch: int = 3,
    max_flows: int = 3,
) -> list[dict]:
    """Trace call-graph flows forward from primary nodes.

    For each primary node that has outgoing calls edges, run BFS to find
    execution chains. Returns a list of flow dicts with seed name and chain.
    """
    from nervx.attention.concepts import _bfs_paths

    if not primary:
        return []

    # Build calls adjacency from all edges (in-memory)
    all_edges = store.get_all_edges()
    calls_adj: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        if e["edge_type"] == "calls":
            calls_adj[e["source_id"]].append(e["target_id"])

    flows: list[dict] = []
    seen_chains: list[set[str]] = []  # for dedup

    for node in primary:
        if len(flows) >= max_flows:
            break
        if node["id"] not in calls_adj:
            continue

        paths = _bfs_paths(calls_adj, node["id"], max_depth, max_paths, max_branch)
        for path in paths:
            if len(path) < 2:
                continue

            path_set = set(path)

            # Deduplicate: skip if >80% overlap with an existing flow
            is_dup = False
            for existing in seen_chains:
                overlap = len(path_set & existing) / max(len(path_set), len(existing))
                if overlap > 0.8:
                    is_dup = True
                    break
            if is_dup:
                continue

            # Resolve node IDs to compact dicts
            chain = []
            for nid in path:
                n = store.get_node(nid)
                if n:
                    chain.append({
                        "id": n["id"],
                        "name": n["name"],
                        "file_path": n["file_path"],
                        "line_start": n["line_start"],
                        "kind": n["kind"],
                    })

            if len(chain) >= 2:
                flows.append({
                    "seed_name": node["name"],
                    "chain": chain,
                })
                seen_chains.append(path_set)

            if len(flows) >= max_flows:
                break

    return flows


def _format_navigate(
    terms: list[str],
    primary: list[dict],
    secondary: list[dict],
    cochange_files: list[dict],
    read_order: list[dict],
    warnings: list[Warning],
    flows: list[dict] | None = None,
    secondary_edge: dict[str, str] | None = None,
) -> str:
    """Format navigate output as token-efficient text."""
    lines = [f"## Navigate: {' '.join(terms)}", ""]
    secondary_edge = secondary_edge or {}

    if primary:
        lines.append("### Relevant Symbols")
        for n in primary:
            doc_preview = ""
            if n["docstring"]:
                doc_preview = f'  "{n["docstring"][:60]}"'
            lines.append(f"  {n['file_path']}:{n['line_start']}  "
                        f"{n['signature'] or n['name']}  [{n['kind']}]{doc_preview}")
        lines.append("")

    if flows:
        lines.append("### Execution Flows")
        for flow in flows:
            chain_names = " -> ".join(c["name"] for c in flow["chain"])
            lines.append(f"  {chain_names}")
        lines.append("")

    if secondary:
        lines.append("### Connected Symbols")
        for n in secondary:
            rel = secondary_edge.get(n["id"], "")
            rel_suffix = f", {rel}" if rel else ""
            lines.append(f"  {n['file_path']}:{n['line_start']}  "
                        f"{n['name']}  [{n['kind']}{rel_suffix}]")
        lines.append("")

    if cochange_files:
        lines.append("### Usually Co-Modified")
        for cf in cochange_files:
            count = cf.get("co_commit_count", 0)
            # Suppress noise: need at least 3 co-commits to show (B5).
            if count and count < 3:
                continue
            count_suffix = f", {count} co-commits" if count else ""
            lines.append(f"  {cf['file_path']}  "
                        f"(coupling={int(cf['coupling_score'] * 100)}%{count_suffix})")
        lines.append("")

    if read_order:
        lines.append("### Suggested Read Order")
        for i, n in enumerate(read_order, 1):
            lines.append(f"  {i}. {n['file_path']}:{n['line_start']}-{n['line_end']}  "
                        f"({n['name']})")
        lines.append("")

    if warnings:
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"  ⚠ {w.message}")
        lines.append("")

    return "\n".join(lines)


# ── Find command ──────────────────────────────────────────────────

def find(
    store: GraphStore,
    kind: str | None = None,
    tag: str | None = None,
    no_tests: bool = False,
    no_direct_tests: bool = False,
    importance_gt: float | None = None,
    cross_module: bool = False,
    dead: bool = False,
    exclude_category: list[str] | None = None,
    include_category: list[str] | None = None,
) -> list[dict]:
    """Structural query over the graph.

    ``no_tests``         — exclude symbols that are directly *or* transitively
                           reachable from a test-tagged node (3-hop BFS). This
                           is the default "what code lacks any test coverage?"
                           semantics users expect.
    ``no_direct_tests``  — stricter: exclude only symbols with a direct test
                           call edge. Use when you want to audit wrappers too.
    ``exclude_category`` — drop nodes whose ``category:<name>`` tag is in the set.
    ``include_category`` — keep only nodes whose ``category:<name>`` is in the set.
    """
    from nervx.attention.test_coverage import test_coverage_for

    conditions = ["1=1"]
    params: list = []

    if kind:
        conditions.append("kind = ?")
        params.append(kind)

    if importance_gt is not None:
        conditions.append("importance > ?")
        params.append(importance_gt)

    where = " AND ".join(conditions)
    rows = store.conn.execute(
        f"SELECT * FROM nodes WHERE {where} ORDER BY importance DESC",
        params,
    ).fetchall()

    # Pre-compute incoming usage map for dead code detection
    incoming_usage: dict[str, int] | None = None
    classes_with_bases: set[str] = set()
    if dead:
        incoming_usage = _build_incoming_usage(store)
        classes_with_bases = _build_class_has_bases(store)

    exclude_set = {c.strip().lower() for c in (exclude_category or []) if c.strip()}
    include_set = {c.strip().lower() for c in (include_category or []) if c.strip()}

    results = []
    for r in rows:
        node = dict(r)
        tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]

        if tag and tag not in tags:
            continue

        # C17 category filters
        if exclude_set or include_set:
            node_category = ""
            for t in tags:
                if isinstance(t, str) and t.startswith("category:"):
                    node_category = t.split(":", 1)[1].lower()
                    break
            if exclude_set and node_category in exclude_set:
                continue
            if include_set and node_category not in include_set:
                continue

        # A test node is trivially "tested" by itself — skip it from either mode.
        if (no_tests or no_direct_tests) and "test" in tags:
            continue

        if no_direct_tests:
            cov = test_coverage_for(store, node["id"], max_hops=3)
            if cov["direct_count"] > 0:
                continue
        elif no_tests:
            cov = test_coverage_for(store, node["id"], max_hops=3)
            if cov["direct_count"] > 0 or cov["transitive"]:
                continue

        if cross_module:
            cross = store.get_cross_module_edges(node["id"])
            if cross == 0:
                continue

        if dead:
            if not _is_dead(node, tags, incoming_usage, classes_with_bases):
                continue

        results.append(node)

    return results


# Tags that indicate a node is invoked by frameworks/runtime, not static callers
_ALIVE_TAGS = frozenset({
    "entrypoint", "route_handler", "test", "callback", "dunder",
    "property", "validator", "override", "overload", "hook",
    "exported", "abstract", "classmethod", "static",
})


def _build_incoming_usage(store: GraphStore) -> dict[str, int]:
    """Build map of node_id -> incoming usage edge count (bulk, O(E))."""
    usage_types = {"calls", "imports", "inherits", "instantiates"}
    all_edges = store.get_all_edges()
    incoming: dict[str, int] = {}
    for e in all_edges:
        if e["edge_type"] in usage_types:
            incoming[e["target_id"]] = incoming.get(e["target_id"], 0) + 1
    return incoming


def _build_class_has_bases(store: GraphStore) -> set[str]:
    """Build set of class node IDs that have base classes (potential overrides)."""
    all_nodes = store.get_all_nodes()
    classes_with_bases: set[str] = set()
    for node in all_nodes:
        if node["kind"] == "class":
            tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]
            for t in tags:
                if t.startswith("extends:"):
                    classes_with_bases.add(node["id"])
                    break
    return classes_with_bases


def _is_dead(
    node: dict,
    tags: list,
    incoming_usage: dict[str, int],
    classes_with_bases: set[str],
) -> bool:
    """Check if a node qualifies as dead code (conservative)."""
    from nervx.instinct.frameworks import is_framework_entrypoint

    # Skip file nodes
    if node["kind"] == "file":
        return False

    # Framework-aware filter — covers dunder, route handlers, fixtures,
    # subclass overrides, lifecycle methods, etc.
    node_for_framework = dict(node)
    node_for_framework["tags"] = tags
    if is_framework_entrypoint(node_for_framework):
        return False

    # Belt-and-braces: legacy _ALIVE_TAGS check in case new tags slip in.
    if _ALIVE_TAGS & set(tags):
        return False

    name = node["name"]

    # Skip main functions
    if name == "main":
        return False

    # Skip methods on classes that inherit from something —
    # they could be overrides called via polymorphism
    if node["kind"] == "method" and node.get("parent_id", "") in classes_with_bases:
        return False

    # Skip public methods of classes (could be API surface)
    # Only flag private methods/functions as dead
    if node["kind"] == "method" and not name.startswith("_"):
        return False

    # Skip data_model classes (Pydantic models, dataclasses etc.)
    if node["kind"] == "class" and "data_model" in tags:
        return False

    # Dead = zero incoming usage edges
    return incoming_usage.get(node["id"], 0) == 0


# ── Blast radius query ────────────────────────────────────────────

def blast_radius_query(
    store: GraphStore,
    symbol_id: str,
    depth: int = 3,
    pick: int | None = None,
    exclude_category: list[str] | None = None,
    include_category: list[str] | None = None,
) -> str:
    """Format blast radius as text."""
    from nervx.attention.fuzzy import resolve_symbol
    node, error = resolve_symbol(store, symbol_id, pick=pick)
    if node is None:
        return error
    symbol_id = node["id"]
    radius = compute_blast_radius(store, symbol_id, depth)
    name = node["name"]

    exclude_set = {c.strip().lower() for c in (exclude_category or []) if c.strip()}
    include_set = {c.strip().lower() for c in (include_category or []) if c.strip()}

    def _category_ok(nid: str) -> bool:
        if not (exclude_set or include_set):
            return True
        n = store.get_node(nid)
        if not n:
            return True
        cat = _node_category(n.get("tags"))
        if exclude_set and cat in exclude_set:
            return False
        if include_set and cat not in include_set:
            return False
        return True

    if exclude_set or include_set:
        radius.direct = [i for i in radius.direct if _category_ok(i)]
        radius.indirect = [i for i in radius.indirect if _category_ok(i)]
        radius.distant = [i for i in radius.distant if _category_ok(i)]
        radius.total_affected = (
            len(radius.direct) + len(radius.indirect) + len(radius.distant)
        )

    lines = [f"## Blast Radius: {name}", ""]

    def _format_ids(ids: list[str], label: str, prefix: str) -> None:
        if ids:
            lines.append(f"### {label}")
            for nid in ids:
                n = store.get_node(nid)
                if n:
                    lines.append(f"  {prefix} {n['file_path']}:{n['line_start']}  {n['name']}  [{n['kind']}]")
            lines.append("")

    _format_ids(
        radius.direct,
        "Direct callers / subclasses (depth 1 — review before merging)",
        "→",
    )
    _format_ids(radius.indirect, "Indirect (likely affected)", "→→")
    _format_ids(radius.distant, "Distant (possibly affected)", "→→→")

    if radius.temporal:
        lines.append("### Temporal (co-change)")
        for fp in radius.temporal:
            lines.append(f"  ~ {fp}")
        lines.append("")

    if radius.tests:
        lines.append("### Tests to run")
        for tid in radius.tests:
            n = store.get_node(tid)
            if n:
                lines.append(f"  ✓ {n['file_path']}:{n['line_start']}  {n['name']}")
        lines.append("")

    lines.append(f"**Total affected: {radius.total_affected}**")
    return "\n".join(lines)


# ── Diff query ────────────────────────────────────────────────────

def diff_query(store: GraphStore, days: int = 7) -> str:
    """Show structural changes: hotspots, churn."""
    lines = [f"## Structural Diff (last {days} days)", ""]

    # Hotspots
    all_stats = store.get_all_file_stats()
    hotspots = [s for s in all_stats if s.get("commits_7d", 0) >= 1]
    hotspots.sort(key=lambda s: -s.get("commits_7d", 0))

    if hotspots:
        lines.append("### Hotspots")
        for s in hotspots[:10]:
            lines.append(f"  {s['file_path']}  {s['commits_7d']} commits "
                        f"(by {s['primary_author']}, {s['author_count']} authors)")
        lines.append("")
    else:
        lines.append("No hotspots found (no git data or no recent changes).")

    return "\n".join(lines)
