"""GraphStore: all database reads and writes for nervx."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import REVERSE_EDGE_MAP, SCHEMA_SQL


class GraphStore:
    """SQLite-backed graph store for nervx brain database."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self._migrate()
        self._in_batch = False

    def _migrate(self):
        """Lightweight schema migrations for older brain.db files."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(nodes)").fetchall()}
        # Add content_hash column if missing (v0.2).
        if "content_hash" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN content_hash TEXT DEFAULT ''")
            self.conn.commit()
        # Add importance_rank column if missing (v0.2.2 — percentile 0-100).
        if "importance_rank" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN importance_rank INTEGER DEFAULT 0")
            self.conn.commit()
        # C19: commit_ids on cochanges for `cochange --why`.
        cc_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(cochanges)").fetchall()}
        if "commit_ids" not in cc_cols:
            self.conn.execute("ALTER TABLE cochanges ADD COLUMN commit_ids TEXT DEFAULT '[]'")
            self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.conn.close()

    def batch(self):
        """Context manager for batch operations — defers commits until exit."""
        return _BatchContext(self)

    def _commit(self):
        """Commit unless inside a batch."""
        if not self._in_batch:
            self.conn.commit()

    # ── Node operations ──────────────────────────────────────────

    def upsert_node(
        self,
        id: str,
        kind: str,
        name: str,
        file_path: str,
        line_start: int | None = None,
        line_end: int | None = None,
        signature: str = "",
        docstring: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.0,
        importance_rank: int = 0,
        parent_id: str = "",
        content_hash: str = "",
    ):
        self.conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, kind, name, file_path, line_start, line_end,
                signature, docstring, tags, importance, importance_rank,
                parent_id, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id, kind, name, file_path, line_start, line_end,
                signature, docstring, json.dumps(tags or []),
                importance, importance_rank, parent_id, content_hash,
            ),
        )
        self._commit()

    def get_node(self, node_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_nodes_by_file(self, file_path: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_kind(self, kind: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE kind = ?", (kind,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_name(self, name: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE name = ?", (name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_nodes(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM nodes").fetchall()
        return [dict(r) for r in rows]

    def delete_nodes_by_file(self, file_path: str):
        self.conn.execute(
            "DELETE FROM nodes WHERE file_path = ?", (file_path,)
        )
        self._commit()

    # ── Edge operations ──────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict | None = None,
    ):
        meta_json = json.dumps(metadata or {})
        self.conn.execute(
            """INSERT OR REPLACE INTO edges
               (source_id, target_id, edge_type, weight, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, target_id, edge_type, weight, meta_json),
        )
        reverse = REVERSE_EDGE_MAP.get(edge_type)
        if reverse:
            self.conn.execute(
                """INSERT OR REPLACE INTO edges
                   (source_id, target_id, edge_type, weight, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (target_id, source_id, reverse, weight, meta_json),
            )
        self._commit()

    def get_edges_from(self, source_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (source_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_to(self, target_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE target_id = ?", (target_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_edges(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        return [dict(r) for r in rows]

    def get_edges_by_type(self, edge_type: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE edge_type = ?", (edge_type,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_edges_involving_file(self, file_path: str):
        self.conn.execute(
            """DELETE FROM edges WHERE source_id IN
               (SELECT id FROM nodes WHERE file_path = ?)
               OR target_id IN
               (SELECT id FROM nodes WHERE file_path = ?)""",
            (file_path, file_path),
        )
        self._commit()

    def get_in_degree(self, node_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM edges WHERE target_id = ?",
            (node_id,),
        ).fetchone()
        return row["cnt"]

    def get_out_degree(self, node_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM edges WHERE source_id = ?",
            (node_id,),
        ).fetchone()
        return row["cnt"]

    def get_cross_module_edges(self, node_id: str) -> int:
        node = self.get_node(node_id)
        if not node:
            return 0
        node_module = node["file_path"].split("/")[0] if "/" in node["file_path"] else ""
        edges = self.get_edges_from(node_id) + self.get_edges_to(node_id)
        count = 0
        for e in edges:
            other_id = e["target_id"] if e["source_id"] == node_id else e["source_id"]
            other = self.get_node(other_id)
            if other:
                other_module = other["file_path"].split("/")[0] if "/" in other["file_path"] else ""
                if other_module and node_module and other_module != node_module:
                    count += 1
        return count

    # ── Keyword operations ───────────────────────────────────────

    def add_keyword(self, keyword: str, node_id: str, source: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO keywords (keyword, node_id, source)
               VALUES (?, ?, ?)""",
            (keyword, node_id, source),
        )
        self._commit()

    def add_keywords_bulk(self, keywords: list[tuple[str, str, str]]):
        self.conn.executemany(
            """INSERT OR REPLACE INTO keywords (keyword, node_id, source)
               VALUES (?, ?, ?)""",
            keywords,
        )
        self._commit()

    def search_keywords(self, terms: list[str]) -> list[tuple[str, int]]:
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"""SELECT node_id, COUNT(DISTINCT keyword) as match_count
                FROM keywords
                WHERE keyword IN ({placeholders})
                GROUP BY node_id
                ORDER BY match_count DESC""",
            terms,
        ).fetchall()
        return [(row["node_id"], row["match_count"]) for row in rows]

    def search_keywords_weighted(self, terms: list[str]) -> list[tuple[str, float]]:
        """Search keywords with source-aware weighting.

        Name matches score 3x, docstring 2x, tag 2x, file_path 1x.
        Returns (node_id, weighted_score) sorted descending.
        """
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"""SELECT node_id, SUM(
                    CASE source
                        WHEN 'name' THEN 3
                        WHEN 'docstring' THEN 2
                        WHEN 'tag' THEN 2
                        WHEN 'file_path' THEN 1
                        ELSE 1
                    END
                ) as weighted_score
                FROM keywords
                WHERE keyword IN ({placeholders})
                GROUP BY node_id
                ORDER BY weighted_score DESC""",
            terms,
        ).fetchall()
        return [(row["node_id"], row["weighted_score"]) for row in rows]

    def search_keywords_prefix(self, prefixes: list[str], min_len: int = 3) -> list[tuple[str, float]]:
        """Search keywords using prefix matching (LIKE 'prefix%').

        Returns (node_id, weighted_score) sorted descending.
        Prefixes shorter than min_len are skipped.
        """
        valid = [p for p in prefixes if len(p) >= min_len]
        if not valid:
            return []
        # Build OR conditions for LIKE
        conditions = " OR ".join("keyword LIKE ?" for _ in valid)
        params = [f"{p}%" for p in valid]
        rows = self.conn.execute(
            f"""SELECT node_id, SUM(
                    CASE source
                        WHEN 'name' THEN 3
                        WHEN 'docstring' THEN 2
                        WHEN 'tag' THEN 2
                        WHEN 'file_path' THEN 1
                        ELSE 1
                    END
                ) as weighted_score
                FROM keywords
                WHERE {conditions}
                GROUP BY node_id
                ORDER BY weighted_score DESC""",
            params,
        ).fetchall()
        return [(row["node_id"], row["weighted_score"]) for row in rows]

    def search_nodes_by_name(self, terms: list[str]) -> list[tuple[str, int]]:
        """Search node names containing any of the given terms.

        Returns (node_id, match_count) sorted descending.
        """
        if not terms:
            return []
        conditions = " OR ".join("LOWER(name) LIKE ?" for _ in terms)
        params = [f"%{t.lower()}%" for t in terms]
        rows = self.conn.execute(
            f"""SELECT id, name FROM nodes
                WHERE ({conditions}) AND kind != 'file'""",
            params,
        ).fetchall()
        # Score by how many terms matched
        results: dict[str, int] = {}
        for row in rows:
            name_lower = row["name"].lower()
            count = sum(1 for t in terms if t.lower() in name_lower)
            results[row["id"]] = count
        return sorted(results.items(), key=lambda x: -x[1])

    def delete_keywords_for_node(self, node_id: str):
        self.conn.execute(
            "DELETE FROM keywords WHERE node_id = ?", (node_id,)
        )
        self._commit()

    # ── Cochange operations ──────────────────────────────────────

    def upsert_cochange(
        self,
        file_a: str,
        file_b: str,
        co_commit_count: int,
        total_commits_a: int,
        total_commits_b: int,
        last_co_commit: str,
        coupling_score: float,
        commit_ids: list[str] | None = None,
    ):
        a, b = (file_a, file_b) if file_a < file_b else (file_b, file_a)
        self.conn.execute(
            """INSERT OR REPLACE INTO cochanges
               (file_a, file_b, co_commit_count, total_commits_a,
                total_commits_b, last_co_commit, coupling_score, commit_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (a, b, co_commit_count, total_commits_a,
             total_commits_b, last_co_commit, coupling_score,
             json.dumps(commit_ids or [])),
        )
        self._commit()

    def get_cochanges_for_file(self, file_path: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM cochanges
               WHERE file_a = ? OR file_b = ?
               ORDER BY coupling_score DESC""",
            (file_path, file_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_cochanges(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM cochanges ORDER BY coupling_score DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── File stats operations ────────────────────────────────────

    def upsert_file_stats(
        self,
        file_path: str,
        total_commits: int = 0,
        commits_30d: int = 0,
        commits_7d: int = 0,
        last_commit: str = "",
        primary_author: str = "",
        author_count: int = 0,
    ):
        self.conn.execute(
            """INSERT OR REPLACE INTO file_stats
               (file_path, total_commits, commits_30d, commits_7d,
                last_commit, primary_author, author_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_path, total_commits, commits_30d, commits_7d,
             last_commit, primary_author, author_count),
        )
        self._commit()

    def get_file_stats(self, file_path: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM file_stats WHERE file_path = ?", (file_path,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_file_stats(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM file_stats ORDER BY commits_30d DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── File hash operations ─────────────────────────────────────

    def upsert_file_hash(self, file_path: str, content_hash: str, last_parsed: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO file_hashes
               (file_path, content_hash, last_parsed)
               VALUES (?, ?, ?)""",
            (file_path, content_hash, last_parsed),
        )
        self._commit()

    def get_file_hash(self, file_path: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM file_hashes WHERE file_path = ?", (file_path,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_file_hashes(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT file_path, content_hash FROM file_hashes").fetchall()
        return {row["file_path"]: row["content_hash"] for row in rows}

    # ── Concept path operations ──────────────────────────────────

    def add_concept_path(self, id: str, name: str, node_ids: list[str], path_type: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO concept_paths
               (id, name, node_ids, path_type)
               VALUES (?, ?, ?, ?)""",
            (id, name, json.dumps(node_ids), path_type),
        )
        self._commit()

    def get_concept_paths(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM concept_paths").fetchall()
        return [dict(r) for r in rows]

    # ── Pattern operations ───────────────────────────────────────

    def add_pattern(self, node_id: str, pattern: str, detail: dict, implication: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO patterns
               (node_id, pattern, detail, implication)
               VALUES (?, ?, ?, ?)""",
            (node_id, pattern, json.dumps(detail), implication),
        )
        self._commit()

    def get_patterns_for_node(self, node_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM patterns WHERE node_id = ?", (node_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_patterns(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM patterns").fetchall()
        return [dict(r) for r in rows]

    def clear_patterns(self):
        self.conn.execute("DELETE FROM patterns")
        self._commit()

    # ── Contract operations ──────────────────────────────────────

    def add_contract(
        self,
        function_id: str,
        caller_id: str,
        error_handling: str = "none",
        return_usage: str = "ignored",
    ):
        self.conn.execute(
            """INSERT OR REPLACE INTO contracts
               (function_id, caller_id, error_handling, return_usage)
               VALUES (?, ?, ?, ?)""",
            (function_id, caller_id, error_handling, return_usage),
        )
        self._commit()

    def get_contracts_for_function(self, function_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM contracts WHERE function_id = ?", (function_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_contract_conflicts(self) -> list[str]:
        rows = self.conn.execute(
            """SELECT function_id FROM contracts
               GROUP BY function_id
               HAVING COUNT(DISTINCT error_handling) > 1
                  OR COUNT(DISTINCT return_usage) > 1"""
        ).fetchall()
        return [row["function_id"] for row in rows]

    # ── String-ref operations ────────────────────────────────────

    def add_string_ref(
        self,
        literal: str,
        file_path: str,
        line_number: int,
        context: str = "",
    ):
        self.conn.execute(
            """INSERT OR REPLACE INTO string_refs
               (literal, file_path, line_number, context)
               VALUES (?, ?, ?, ?)""",
            (literal, file_path, line_number, context),
        )
        self._commit()

    def add_string_refs_bulk(
        self, rows: list[tuple[str, str, int, str]]
    ):
        self.conn.executemany(
            """INSERT OR REPLACE INTO string_refs
               (literal, file_path, line_number, context)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        self._commit()

    def get_string_refs(self, literal: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM string_refs
               WHERE literal = ?
               ORDER BY file_path, line_number""",
            (literal,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_string_refs_for_file(self, file_path: str):
        self.conn.execute(
            "DELETE FROM string_refs WHERE file_path = ?", (file_path,)
        )
        self._commit()

    # ── Raw import operations ───────────────────────────────────
    #
    # 0.2.6: `raw_imports` is the ground-truth list of every import a file
    # declares (intra-project AND external). The edges table only stores
    # intra-project `imports` edges; this table lets `ask imports <file>`
    # show the full picture — numpy, torch, std, first-party, etc.

    def add_raw_imports_bulk(
        self, rows: list[tuple[str, str, str, int, str]]
    ):
        """rows: (file_path, module_path, imported_names_json, is_from_import, resolved_to_file)"""
        self.conn.executemany(
            """INSERT OR REPLACE INTO raw_imports
               (file_path, module_path, imported_names, is_from_import, resolved_to_file)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._commit()

    def delete_raw_imports_for_file(self, file_path: str):
        self.conn.execute(
            "DELETE FROM raw_imports WHERE file_path = ?", (file_path,)
        )
        self._commit()

    def get_raw_imports(self, file_path: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT module_path, imported_names, is_from_import, resolved_to_file
               FROM raw_imports WHERE file_path = ?
               ORDER BY is_from_import, module_path""",
            (file_path,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Meta operations ──────────────────────────────────────────

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ── Bulk operations ──────────────────────────────────────────

    def clear_all(self):
        for table in [
            "nodes", "edges", "cochanges", "keywords",
            "file_stats", "file_hashes", "concept_paths",
            "patterns", "contracts", "meta", "string_refs",
            "raw_imports",
        ]:
            self.conn.execute(f"DELETE FROM {table}")
        self._commit()

    def clear_file_data(self, file_path: str):
        node_ids = [
            r["id"] for r in self.conn.execute(
                "SELECT id FROM nodes WHERE file_path = ?", (file_path,)
            ).fetchall()
        ]
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            self.conn.execute(
                f"DELETE FROM keywords WHERE node_id IN ({placeholders})",
                node_ids,
            )
            self.conn.execute(
                f"DELETE FROM contracts WHERE function_id IN ({placeholders}) OR caller_id IN ({placeholders})",
                node_ids + node_ids,
            )
            self.conn.execute(
                f"DELETE FROM patterns WHERE node_id IN ({placeholders})",
                node_ids,
            )
            self.conn.execute(
                f"""DELETE FROM edges WHERE source_id IN ({placeholders})
                    OR target_id IN ({placeholders})""",
                node_ids + node_ids,
            )
        self.conn.execute(
            "DELETE FROM nodes WHERE file_path = ?", (file_path,)
        )
        self.conn.execute(
            "DELETE FROM file_hashes WHERE file_path = ?", (file_path,)
        )
        self.conn.execute(
            "DELETE FROM string_refs WHERE file_path = ?", (file_path,)
        )
        self.conn.execute(
            "DELETE FROM raw_imports WHERE file_path = ?", (file_path,)
        )
        self._commit()


class _BatchContext:
    """Defers commits for bulk insert performance."""

    def __init__(self, store: GraphStore):
        self.store = store
        self._was_in_batch = False

    def __enter__(self):
        self._was_in_batch = self.store._in_batch
        if not self._was_in_batch:
            self.store._in_batch = True
            try:
                self.store.conn.execute("BEGIN")
            except Exception:
                pass  # already in a transaction
        return self.store

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_in_batch:
            self.store._in_batch = False
            if exc_type is None:
                try:
                    self.store.conn.commit()
                except Exception:
                    pass
            else:
                try:
                    self.store.conn.rollback()
                except Exception:
                    pass
