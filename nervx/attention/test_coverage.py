"""Test-coverage detection — direct vs transitive.

Single source of truth for "is this symbol tested?" used by peek, ask,
find, and warnings. Answers two questions separately so callers can tell
them apart (see A2 in the v0.2.2 critique — the previous direct-only
check caused false "no tests" reports for wrappers).
"""

from __future__ import annotations

import json
from typing import TypedDict

from nervx.memory.store import GraphStore


class TestCoverage(TypedDict):
    direct_count: int           # test-tagged nodes that directly call this symbol
    transitive: bool             # reachable from a test-tagged node within max_hops
    transitive_via: str           # first test reached, if any (short name)
    transitive_hops: int          # hops to first test, 0 if none


def test_coverage_for(
    store: GraphStore,
    node_id: str,
    max_hops: int = 3,
) -> TestCoverage:
    """Classify test coverage for ``node_id``.

    "Direct": at least one node tagged ``"test"`` has a ``calls`` edge
    whose target is this symbol.

    "Transitive": BFS up the ``called_by`` chain (i.e. "who calls me, who
    calls them") — if any ancestor within ``max_hops`` is tagged ``"test"``,
    the symbol is considered transitively reached by tests. This is still
    a structural check, not runtime coverage, but it catches the common
    "tested through a wrapper" case that direct-only detection misses.
    """
    direct_count = 0
    try:
        row = store.conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id = ?
              AND e.edge_type = 'calls'
              AND n.tags LIKE '%"test"%'
            """,
            (node_id,),
        ).fetchone()
        direct_count = row["cnt"] if row else 0
    except Exception:
        direct_count = 0

    if direct_count > 0:
        return TestCoverage(
            direct_count=direct_count,
            transitive=True,
            transitive_via="",
            transitive_hops=0,
        )

    # BFS up called_by edges (who calls me, transitively). Every ancestor
    # tagged "test" counts as a hit.
    visited: set[str] = {node_id}
    frontier: list[tuple[str, int]] = [(node_id, 0)]
    while frontier:
        nid, depth = frontier.pop(0)
        if depth >= max_hops:
            continue
        # Use called_by edges from nid → callers of nid.
        for e in store.get_edges_from(nid):
            if e["edge_type"] != "called_by":
                continue
            caller_id = e["target_id"]
            if caller_id in visited:
                continue
            visited.add(caller_id)
            caller = store.get_node(caller_id)
            if not caller:
                continue
            caller_tags = caller.get("tags") or "[]"
            try:
                tags = json.loads(caller_tags) if isinstance(caller_tags, str) else list(caller_tags)
            except (TypeError, ValueError):
                tags = []
            if "test" in tags:
                return TestCoverage(
                    direct_count=0,
                    transitive=True,
                    transitive_via=caller["name"],
                    transitive_hops=depth + 1,
                )
            frontier.append((caller_id, depth + 1))

    return TestCoverage(
        direct_count=0,
        transitive=False,
        transitive_via="",
        transitive_hops=0,
    )


def format_coverage_hint(cov: TestCoverage) -> str:
    """One-line hint suitable for peek output."""
    if cov["direct_count"] > 0:
        plural = "s" if cov["direct_count"] != 1 else ""
        return f"{cov['direct_count']} direct test ref{plural}"
    if cov["transitive"]:
        return (
            f"no direct tests; reachable from test `{cov['transitive_via']}` "
            f"({cov['transitive_hops']} hop"
            f"{'s' if cov['transitive_hops'] != 1 else ''})"
        )
    return "no test coverage found (direct + 3-hop transitive)"
