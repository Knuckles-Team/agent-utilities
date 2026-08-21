"""Contract-test package marker.

Without this file, pytest's ``prepend`` import mode makes ``tests/contract``
the package root, so ``tests/contract/http/test_transport_parity.py`` imports
as ``http.test_transport_parity`` -- shadowing the STDLIB ``http`` package and
failing collection outright:

    ModuleNotFoundError: No module named 'http.test_transport_parity'

A collection error aborts the whole run, so this one missing file took the
entire CI suite down (``pytest -q -n auto --dist loadfile`` over
``testpaths = tests``) while every local gate stayed green -- the pre-push
``pytest`` hook only names ``tests/unit tests/integration tests/retrieval``.

``tests/`` and ``tests/contract/http/`` were already packages; only this
middle level was missing, which is exactly why the resolved name landed on
``http`` rather than ``tests.contract.http``.
"""
