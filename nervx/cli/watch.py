"""Watch mode: auto-update nervx on file changes."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from nervx.build import ALL_EXTENSIONS, EXCLUDE_DIRS, EXCLUDE_FILES


def _should_handle(file_path: str, repo_root: str) -> bool:
    """Check if a changed file should trigger an update."""
    # Must have a supported extension
    if not any(file_path.endswith(ext) for ext in ALL_EXTENSIONS):
        return False

    # Must not be in an excluded directory
    try:
        rel = os.path.relpath(file_path, repo_root)
    except ValueError:
        return False
    parts = Path(rel).parts
    for part in parts[:-1]:  # directories only, not the filename
        if part in EXCLUDE_DIRS or part.endswith(".egg-info"):
            return False

    # Must not be an excluded file
    basename = os.path.basename(file_path)
    if basename in EXCLUDE_FILES:
        return False

    return True


class _ChangeCollector:
    """Thread-safe collector for file change events."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._changes: set[str] = set()
        self._lock = threading.Lock()

    def add(self, path: str) -> None:
        """Record a changed file path (if it passes filters)."""
        if _should_handle(path, self.repo_root):
            with self._lock:
                self._changes.add(path)

    def get_and_clear(self) -> set[str]:
        """Return accumulated changes and reset. Thread-safe."""
        with self._lock:
            changes = self._changes.copy()
            self._changes.clear()
        return changes


def run_watch(repo_root: str, db_path: str, debounce: float = 2.0) -> None:
    """Watch repo for file changes, auto-trigger incremental update.

    Blocks until Ctrl+C.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print(
            "Watch mode requires 'watchdog'. Install with:\n"
            "  pip install watchdog\n"
            "Or: pip install nervx[watch]",
            file=sys.stderr,
        )
        sys.exit(1)

    collector = _ChangeCollector(repo_root)
    updating = False

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                collector.add(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                collector.add(event.src_path)

        def on_deleted(self, event):
            if not event.is_directory:
                collector.add(event.src_path)

    observer = Observer()
    observer.schedule(_Handler(), repo_root, recursive=True)
    observer.start()

    print(f"Watching {repo_root} for changes (Ctrl+C to stop)...")

    try:
        while True:
            time.sleep(debounce)

            changes = collector.get_and_clear()
            if not changes or updating:
                continue

            updating = True
            n_changes = len(changes)
            print(f"\nDetected {n_changes} change{'s' if n_changes > 1 else ''}, updating...")

            try:
                from nervx.build import incremental_update
                from nervx.attention.briefing import generate_briefing, inject_claude_md
                from nervx.memory.store import GraphStore

                incremental_update(repo_root, db_path)

                # Regenerate NERVX.md
                store = GraphStore(db_path)
                briefing = generate_briefing(store, repo_root)
                nervx_md = os.path.join(repo_root, "NERVX.md")
                with open(nervx_md, "w", encoding="utf-8") as f:
                    f.write(briefing)
                store.close()

                # Update CLAUDE.md
                inject_claude_md(repo_root)

                print("Update complete.")
            except Exception as e:
                print(f"Update failed: {e}", file=sys.stderr)
            finally:
                updating = False

    except KeyboardInterrupt:
        print("\nStopping watch...")
        observer.stop()

    observer.join()
