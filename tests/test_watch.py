"""Tests for watch mode file filtering and change collection."""

import os

from nervx.cli.watch import _should_handle, _ChangeCollector


REPO_ROOT = "/fake/repo"


def test_should_handle_python_file():
    assert _should_handle("/fake/repo/src/main.py", REPO_ROOT) is True


def test_should_handle_js_file():
    assert _should_handle("/fake/repo/app.js", REPO_ROOT) is True


def test_should_handle_ts_file():
    assert _should_handle("/fake/repo/src/index.tsx", REPO_ROOT) is True


def test_should_reject_non_source():
    assert _should_handle("/fake/repo/README.md", REPO_ROOT) is False
    assert _should_handle("/fake/repo/data.json", REPO_ROOT) is False
    assert _should_handle("/fake/repo/image.png", REPO_ROOT) is False


def test_should_reject_excluded_dirs():
    assert _should_handle("/fake/repo/node_modules/pkg/index.js", REPO_ROOT) is False
    assert _should_handle("/fake/repo/__pycache__/module.py", REPO_ROOT) is False
    assert _should_handle("/fake/repo/.git/objects/abc", REPO_ROOT) is False
    assert _should_handle("/fake/repo/.venv/lib/site.py", REPO_ROOT) is False
    assert _should_handle("/fake/repo/.nervx/brain.db", REPO_ROOT) is False


def test_should_reject_excluded_files():
    assert _should_handle("/fake/repo/setup.py", REPO_ROOT) is False
    assert _should_handle("/fake/repo/conftest.py", REPO_ROOT) is False
    assert _should_handle("/fake/repo/package-lock.json", REPO_ROOT) is False


def test_should_handle_nested_source():
    assert _should_handle("/fake/repo/src/components/Button.tsx", REPO_ROOT) is True
    assert _should_handle("/fake/repo/lib/utils/helper.go", REPO_ROOT) is True


def test_change_collector_filters():
    collector = _ChangeCollector(REPO_ROOT)
    collector.add("/fake/repo/src/main.py")
    collector.add("/fake/repo/README.md")  # rejected
    collector.add("/fake/repo/node_modules/a.js")  # rejected
    collector.add("/fake/repo/src/utils.py")

    changes = collector.get_and_clear()
    assert len(changes) == 2
    assert "/fake/repo/src/main.py" in changes
    assert "/fake/repo/src/utils.py" in changes


def test_change_collector_clear():
    collector = _ChangeCollector(REPO_ROOT)
    collector.add("/fake/repo/a.py")
    changes = collector.get_and_clear()
    assert len(changes) == 1

    # Second call should be empty
    changes2 = collector.get_and_clear()
    assert len(changes2) == 0


def test_change_collector_dedup():
    collector = _ChangeCollector(REPO_ROOT)
    collector.add("/fake/repo/a.py")
    collector.add("/fake/repo/a.py")
    collector.add("/fake/repo/a.py")
    changes = collector.get_and_clear()
    assert len(changes) == 1
