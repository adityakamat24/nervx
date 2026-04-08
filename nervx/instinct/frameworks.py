"""Framework-aware liveness detection.

Recognizes symbols that are called by frameworks at runtime but have zero
static in-degree. Used by `find --dead` to avoid flagging obvious entry points.
"""

from __future__ import annotations

import json

# Decorator names (or suffixes) that make a symbol a framework entry point.
# Matched against `decorator:<name>` tags added by the parser.
FRAMEWORK_DECORATORS: frozenset[str] = frozenset({
    # FastAPI / Starlette
    "app.get", "app.post", "app.put", "app.delete", "app.patch",
    "app.options", "app.head", "app.websocket", "app.on_event",
    "router.get", "router.post", "router.put", "router.delete",
    "router.patch", "router.websocket",
    # Flask
    "app.route", "blueprint.route", "bp.route",
    # Django
    "admin.register", "receiver",
    # Click / Typer
    "click.command", "click.group", "click.option", "click.argument",
    "app.command", "typer.command",
    # Pytest
    "pytest.fixture", "fixture", "pytest.mark.parametrize",
    "pytest.mark.asyncio",
    # Celery
    "app.task", "shared_task", "celery.task",
    # General
    "abstractmethod",
})

# Tags set by the parser that independently indicate a framework callsite.
_LIVE_TAGS: frozenset[str] = frozenset({
    "entrypoint", "route_handler", "test", "callback", "dunder",
    "property", "validator", "override", "overload", "hook",
    "exported", "abstract", "classmethod", "static",
})

# Lifecycle method names that frameworks call by convention.
_LIFECYCLE_METHODS: frozenset[str] = frozenset({
    "setUp", "tearDown", "setUpClass", "tearDownClass",
    "setup_method", "teardown_method", "setup_class", "teardown_class",
    "on_startup", "on_shutdown", "lifespan",
    "ready", "apps", "save", "delete", "clean", "full_clean",
    "get_queryset", "get_serializer_class", "perform_create",
    "dispatch", "__init_subclass__",
})


def _tags_as_list(raw) -> list[str]:
    """Normalize tags that may arrive as either a JSON string or a list."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


def is_framework_entrypoint(node: dict) -> bool:
    """Return True if `node` is plausibly called by a framework at runtime.

    Conservative: prefer false negatives to false positives. A false positive
    here means flagging live code as dead, which erodes trust in `find --dead`.
    """
    tags = _tags_as_list(node.get("tags", []))
    name = node.get("name", "") or ""

    for tag in tags:
        if tag in _LIVE_TAGS:
            return True

        # Decorator tags from the parser: "decorator:app.route", "decorator:pytest.fixture"
        if tag.startswith("decorator:"):
            dec = tag[len("decorator:"):]
            dec_simple = dec.split(".")[-1] if "." in dec else dec
            for known in FRAMEWORK_DECORATORS:
                # Match full dotted name ("app.route") OR the trailing segment
                # ("route" against "app.route", "router.get" etc.) so we catch
                # aliased apps like `api.get` without enumerating them all.
                if dec == known or dec.endswith("." + known):
                    return True
                known_simple = known.split(".")[-1] if "." in known else known
                if dec_simple == known_simple:
                    return True

        # Subclassing — likely polymorphic, can be called via base-class dispatch.
        if tag.startswith("extends:"):
            return True

    # Dunder methods are called implicitly by the Python runtime.
    if name.startswith("__") and name.endswith("__"):
        return True

    # Pytest / unittest convention.
    if name.startswith("test_") or name.startswith("Test"):
        return True

    if name in _LIFECYCLE_METHODS:
        return True

    # Django / DRF naming conventions that imply runtime registration.
    if name.endswith(("View", "Serializer", "Admin", "Form", "ViewSet")):
        return True

    return False
