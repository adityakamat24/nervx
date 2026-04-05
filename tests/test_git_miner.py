"""Tests for git history mining."""

import os
import subprocess
import tempfile

import pytest

from nervx.memory.store import GraphStore
from nervx.perception.git_miner import GitMiner, is_git_repo


def _run_git(cwd, *args):
    subprocess.run(
        ["git"] + list(args), cwd=cwd,
        capture_output=True, text=True, check=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    """Create a temp git repo with known commit history."""
    repo = str(tmp_path)
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@test.com")
    _run_git(repo, "config", "user.name", "Test")

    # Commit 1: a.py and b.py together
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    _run_git(repo, "add", "a.py", "b.py")
    _run_git(repo, "commit", "-m", "commit 1")

    # Commit 2: a.py and b.py together again
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "b.py").write_text("y = 3\n")
    _run_git(repo, "add", "a.py", "b.py")
    _run_git(repo, "commit", "-m", "commit 2")

    # Commit 3: only c.py (single file, no co-change data)
    (tmp_path / "c.py").write_text("z = 1\n")
    _run_git(repo, "add", "c.py")
    _run_git(repo, "commit", "-m", "commit 3")

    # Commit 4: a.py and c.py
    (tmp_path / "a.py").write_text("x = 3\n")
    (tmp_path / "c.py").write_text("z = 2\n")
    _run_git(repo, "add", "a.py", "c.py")
    _run_git(repo, "commit", "-m", "commit 4")

    return repo


def test_is_git_repo(git_repo, tmp_path_factory):
    assert is_git_repo(git_repo)
    non_repo = str(tmp_path_factory.mktemp("not_a_repo"))
    assert not is_git_repo(non_repo)


def test_file_stats(git_repo):
    store = GraphStore(":memory:")
    miner = GitMiner(git_repo)
    miner.mine(store)

    stats_a = store.get_file_stats("a.py")
    assert stats_a is not None
    assert stats_a["total_commits"] == 3  # commits 1, 2, 4

    stats_b = store.get_file_stats("b.py")
    assert stats_b is not None
    assert stats_b["total_commits"] == 2  # commits 1, 2

    stats_c = store.get_file_stats("c.py")
    assert stats_c is not None
    assert stats_c["total_commits"] == 2  # commits 3, 4

    assert stats_a["primary_author"] == "test@test.com"
    assert stats_a["author_count"] == 1

    store.close()


def test_cochanges(git_repo):
    store = GraphStore(":memory:")
    miner = GitMiner(git_repo)
    miner.mine(store)

    # a.py and b.py co-changed in commits 1 and 2 (count=2)
    cochanges_a = store.get_cochanges_for_file("a.py")
    ab_pair = [c for c in cochanges_a if c["file_b"] == "b.py" or c["file_a"] == "b.py"]
    assert len(ab_pair) == 1
    assert ab_pair[0]["co_commit_count"] == 2

    # a.py and c.py only co-changed once (commit 4), so should be filtered out (count < 2)
    ac_pair = [c for c in cochanges_a if c["file_b"] == "c.py" or c["file_a"] == "c.py"]
    assert len(ac_pair) == 0

    store.close()


def test_single_file_commit_no_cochange(git_repo):
    """Single-file commits shouldn't produce co-change data."""
    store = GraphStore(":memory:")
    miner = GitMiner(git_repo)
    miner.mine(store)

    # c.py was alone in commit 3 - check it doesn't pair with itself
    cochanges_c = store.get_cochanges_for_file("c.py")
    # Only from commit 4 (with a.py), but count=1 so filtered
    assert len(cochanges_c) == 0

    store.close()


def test_non_py_files_ignored(git_repo):
    """Non-.py files should not appear in stats."""
    # Add a .txt file
    txt_path = os.path.join(git_repo, "readme.txt")
    with open(txt_path, "w") as f:
        f.write("hello\n")
    _run_git(git_repo, "add", "readme.txt")
    _run_git(git_repo, "commit", "-m", "add readme")

    store = GraphStore(":memory:")
    miner = GitMiner(git_repo)
    miner.mine(store)

    assert store.get_file_stats("readme.txt") is None

    store.close()
