"""Language metadata and extension registry.

nervx ships with per-language parsers in `lang_*.py` modules. This module is
the single source of truth for which file extensions are tracked and which
language each one belongs to — used by the build walker, the peek/tree/ask
commands, and anything that needs to route on language.

To add a new language:
  1. Write a `lang_<name>.py` parser returning a `ParseResult`.
  2. Wire it into `parser.parse_file` dispatch.
  3. Register it here with `_register(LanguageConfig(...))`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LanguageConfig:
    """Metadata about a supported language."""

    name: str
    extensions: list[str]
    # Tree-sitter grammar name (for tree-sitter-language-pack lookup, if used).
    ts_name: str = ""
    # Line comment token — used by string-refs to skip commented strings.
    line_comment: str = "#"
    # Does this language support decorators/annotations?
    has_decorators: bool = False
    # Does this language have top-level functions (vs. method-only like Java)?
    has_top_level_functions: bool = True


LANGUAGES: dict[str, LanguageConfig] = {}


def _register(config: LanguageConfig) -> None:
    for ext in config.extensions:
        LANGUAGES[ext.lower()] = config


_register(LanguageConfig(
    name="python",
    extensions=[".py"],
    ts_name="python",
    line_comment="#",
    has_decorators=True,
))

_register(LanguageConfig(
    name="javascript",
    extensions=[".js", ".jsx", ".mjs", ".cjs"],
    ts_name="javascript",
    line_comment="//",
    has_decorators=True,
))

_register(LanguageConfig(
    name="typescript",
    extensions=[".ts"],
    ts_name="typescript",
    line_comment="//",
    has_decorators=True,
))

_register(LanguageConfig(
    name="tsx",
    extensions=[".tsx"],
    ts_name="tsx",
    line_comment="//",
    has_decorators=True,
))

_register(LanguageConfig(
    name="java",
    extensions=[".java"],
    ts_name="java",
    line_comment="//",
    has_decorators=True,
    has_top_level_functions=False,
))

_register(LanguageConfig(
    name="go",
    extensions=[".go"],
    ts_name="go",
    line_comment="//",
))

_register(LanguageConfig(
    name="rust",
    extensions=[".rs"],
    ts_name="rust",
    line_comment="//",
    has_decorators=True,
))

_register(LanguageConfig(
    name="c",
    extensions=[".c", ".h"],
    ts_name="c",
    line_comment="//",
    has_top_level_functions=True,
))

_register(LanguageConfig(
    name="cpp",
    extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hh"],
    ts_name="cpp",
    line_comment="//",
))

_register(LanguageConfig(
    name="c_sharp",
    extensions=[".cs"],
    ts_name="csharp",
    line_comment="//",
    has_decorators=True,
    has_top_level_functions=False,
))

_register(LanguageConfig(
    name="ruby",
    extensions=[".rb"],
    ts_name="ruby",
    line_comment="#",
))


def get_language_config(file_path: str) -> LanguageConfig | None:
    """Return the LanguageConfig for a file, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGES.get(ext)


def get_supported_extensions() -> set[str]:
    """Return every tracked file extension (lowercased, with leading dot)."""
    return set(LANGUAGES.keys())


def get_language_name(file_path: str) -> str:
    """Return the language name ("python", "javascript", ...) or '' if unknown."""
    cfg = get_language_config(file_path)
    return cfg.name if cfg else ""
