"""Git history mining: file stats and co-modification analysis."""

from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nervx.memory.store import GraphStore


_GH_NOREPLY_RE = re.compile(
    r"^(\d+\+)?(?P<handle>[A-Za-z0-9][A-Za-z0-9._-]*)@users\.noreply\.github\.com$"
)


def _normalize_author(raw: str) -> str:
    """Clean up a git ``Name <email>`` string.

    - Turns the GitHub noreply pattern ``1234567+alice@users.noreply.github.com``
      into ``@alice`` so authorship reads naturally in diff/briefing output.
    - Collapses whitespace.
    """
    raw = raw.strip()
    if not raw:
        return raw
    # Pull out the email (last <...> token) and attempt handle extraction.
    if "<" in raw and raw.endswith(">"):
        name_part = raw[: raw.rfind("<")].strip()
        email = raw[raw.rfind("<") + 1 : -1].strip()
        m = _GH_NOREPLY_RE.match(email)
        if m:
            handle = m.group("handle")
            # Prefer "Name (@handle)" when name is present, "@handle" otherwise.
            return f"{name_part} (@{handle})" if name_part else f"@{handle}"
        if not name_part:
            return email
    return raw

# All supported source file extensions (must stay in sync with build.ALL_EXTENSIONS)
_ALL_EXTENSIONS = (
    ".py",
    ".js", ".jsx", ".ts", ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh",
    ".cs",
    ".rb",
)


@dataclass
class Commit:
    hash: str
    author: str
    date: str  # ISO format
    files: list[str]


def is_git_repo(path: str) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeError):
        return False


class GitMiner:
    """Mines git history for file stats and co-modification data."""

    def __init__(
        self,
        repo_root: str,
        max_commits: int = 1000,
        extensions: tuple[str, ...] = _ALL_EXTENSIONS,
    ):
        self.repo_root = repo_root
        self.max_commits = max_commits
        self.extensions = extensions

    def mine(self, store: GraphStore):
        """Run git mining and store results."""
        commits = self._parse_git_log()
        if not commits:
            return

        file_stats = self._compute_file_stats(commits)
        for fp, stats in file_stats.items():
            store.upsert_file_stats(
                file_path=fp,
                total_commits=stats["total_commits"],
                commits_30d=stats["commits_30d"],
                commits_7d=stats["commits_7d"],
                last_commit=stats["last_commit"],
                primary_author=stats["primary_author"],
                author_count=stats["author_count"],
            )

        cochanges = self._compute_cochanges(commits, file_stats)
        for (fa, fb), data in cochanges.items():
            store.upsert_cochange(
                file_a=fa,
                file_b=fb,
                co_commit_count=data["count"],
                total_commits_a=file_stats[fa]["total_commits"],
                total_commits_b=file_stats[fb]["total_commits"],
                last_co_commit=data["last"],
                coupling_score=data["coupling"],
                commit_ids=data.get("commit_ids", []),
            )

    def _parse_git_log(self) -> list[Commit]:
        """Run git log and parse the output into Commit objects.

        Git log output routinely contains non-ASCII bytes (author names,
        commit messages, file paths). On Windows, subprocess defaults to
        cp1252 which crashes on anything outside latin-1. We force utf-8
        with ``errors='replace'`` so mining can't silently die on a single
        weird byte — the worst case is a replacement character in an
        author name, which is strictly better than no git data at all.
        """
        try:
            result = subprocess.run(
                [
                    "git", "log",
                    f"--max-count={self.max_commits}",
                    # %aN uses .mailmap when present and falls back to the
                    # author name — cleaner attribution than raw emails.
                    "--format=COMMIT|%H|%aN <%ae>|%aI",
                    "--name-only",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeError):
            return []

        if result.returncode != 0 or not result.stdout:
            return []

        commits: list[Commit] = []
        current: Commit | None = None

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("COMMIT|"):
                if current is not None:
                    commits.append(current)
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    author = _normalize_author(parts[2])
                    current = Commit(
                        hash=parts[1], author=author,
                        date=parts[3], files=[],
                    )
                else:
                    current = None
            elif current is not None:
                # This is a file path
                fp = line.replace("\\", "/")
                if any(fp.endswith(ext) for ext in self.extensions):
                    current.files.append(fp)

        if current is not None:
            commits.append(current)

        return commits

    def _compute_file_stats(self, commits: list[Commit]) -> dict[str, dict]:
        """Compute per-file statistics from commits."""
        now = datetime.now(timezone.utc)
        d30 = now - timedelta(days=30)
        d7 = now - timedelta(days=7)

        stats: dict[str, dict] = {}

        for commit in commits:
            try:
                cdate = datetime.fromisoformat(commit.date)
            except ValueError:
                cdate = now

            for fp in commit.files:
                if fp not in stats:
                    stats[fp] = {
                        "total_commits": 0,
                        "commits_30d": 0,
                        "commits_7d": 0,
                        "last_commit": commit.date,
                        "authors": Counter(),
                        "primary_author": "",
                        "author_count": 0,
                    }

                s = stats[fp]
                s["total_commits"] += 1
                s["authors"][commit.author] += 1

                if cdate >= d30:
                    s["commits_30d"] += 1
                if cdate >= d7:
                    s["commits_7d"] += 1

        # Finalize
        for fp, s in stats.items():
            if s["authors"]:
                s["primary_author"] = s["authors"].most_common(1)[0][0]
                s["author_count"] = len(s["authors"])
            del s["authors"]

        return stats

    def _compute_cochanges(
        self, commits: list[Commit], file_stats: dict[str, dict],
    ) -> dict[tuple[str, str], dict]:
        """Compute co-modification matrix from commits.

        Also collects the short commit hashes behind each pair so
        ``nervx cochange --why`` can point at specific commits.
        """
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_last: dict[tuple[str, str], str] = {}
        pair_commits: dict[tuple[str, str], list[str]] = {}

        for commit in commits:
            files = commit.files
            # Only consider commits with 2-15 changed files
            if len(files) < 2 or len(files) > 15:
                continue

            short = commit.hash[:8]
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    pair = (fa, fb) if fa < fb else (fb, fa)
                    pair_counts[pair] += 1
                    pair_commits.setdefault(pair, []).append(short)
                    if pair not in pair_last:
                        pair_last[pair] = commit.date
                    else:
                        # Keep the most recent
                        if commit.date > pair_last[pair]:
                            pair_last[pair] = commit.date

        # Filter and compute coupling scores
        result: dict[tuple[str, str], dict] = {}
        for pair, count in pair_counts.items():
            if count < 2:
                continue
            fa, fb = pair
            max_commits = max(
                file_stats.get(fa, {}).get("total_commits", 1),
                file_stats.get(fb, {}).get("total_commits", 1),
            )
            coupling = count / max_commits if max_commits > 0 else 0
            if coupling < 0.2:
                continue

            # Cap stored hashes — anything above ~20 is just noise for display.
            commit_list = pair_commits.get(pair, [])[:20]
            result[pair] = {
                "count": count,
                "coupling": round(coupling, 3),
                "last": pair_last.get(pair, ""),
                "commit_ids": commit_list,
            }

        return result
