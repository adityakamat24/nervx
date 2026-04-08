"""SQLite schema for nervx brain database."""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    signature TEXT,
    docstring TEXT,
    tags TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.0,
    parent_id TEXT DEFAULT '',
    content_hash TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS cochanges (
    file_a TEXT NOT NULL,
    file_b TEXT NOT NULL,
    co_commit_count INTEGER DEFAULT 0,
    total_commits_a INTEGER DEFAULT 0,
    total_commits_b INTEGER DEFAULT 0,
    last_co_commit TEXT,
    coupling_score REAL DEFAULT 0.0,
    PRIMARY KEY (file_a, file_b)
);

CREATE TABLE IF NOT EXISTS keywords (
    keyword TEXT NOT NULL,
    node_id TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (keyword, node_id, source)
);

CREATE TABLE IF NOT EXISTS file_stats (
    file_path TEXT PRIMARY KEY,
    total_commits INTEGER DEFAULT 0,
    commits_30d INTEGER DEFAULT 0,
    commits_7d INTEGER DEFAULT 0,
    last_commit TEXT,
    primary_author TEXT,
    author_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_parsed TEXT
);

CREATE TABLE IF NOT EXISTS concept_paths (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    node_ids TEXT DEFAULT '[]',
    path_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    node_id TEXT NOT NULL,
    pattern TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    implication TEXT DEFAULT '',
    PRIMARY KEY (node_id, pattern)
);

CREATE TABLE IF NOT EXISTS contracts (
    function_id TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    error_handling TEXT DEFAULT 'none',
    return_usage TEXT DEFAULT 'ignored',
    arguments TEXT DEFAULT '{}',
    PRIMARY KEY (function_id, caller_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS string_refs (
    literal TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    context TEXT DEFAULT '',
    PRIMARY KEY (literal, file_path, line_number)
);
CREATE INDEX IF NOT EXISTS idx_string_refs_literal ON string_refs(literal);
CREATE INDEX IF NOT EXISTS idx_string_refs_file ON string_refs(file_path);

CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_keywords_node ON keywords(node_id);
CREATE INDEX IF NOT EXISTS idx_cochanges_filea ON cochanges(file_a);
CREATE INDEX IF NOT EXISTS idx_cochanges_fileb ON cochanges(file_b);
CREATE INDEX IF NOT EXISTS idx_file_stats_commits ON file_stats(commits_30d DESC);
"""

REVERSE_EDGE_MAP = {
    "calls": "called_by",
    "called_by": "calls",
    "imports": "imported_by",
    "imported_by": "imports",
    "inherits": "inherited_by",
    "inherited_by": "inherits",
    "instantiates": None,
}
