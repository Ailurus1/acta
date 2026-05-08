from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    hook_items = [item for item in items if "tests/test_hook.py" in item.nodeid]
    other_items = [item for item in items if "tests/test_hook.py" not in item.nodeid]
    items[:] = hook_items + other_items


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    if call.when != "call":
        return
    if "tests/test_hook.py" in item.nodeid and call.excinfo is not None:
        item.session.config._acta_hook_failed = True


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "tests/test_analyzer.py" in item.nodeid and getattr(
        item.session.config, "_acta_hook_failed", False
    ):
        pytest.skip(
            "Skipping analyzer tests because at least one test_hook test failed."
        )
