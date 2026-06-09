"""Shared base types for model provider clients.

`ProviderError` is the common error type every provider client raises. The older
`SiliconFlowError` is kept as a subclass so existing `except SiliconFlowError`
sites keep working while new code can catch `ProviderError` across all providers.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Raised by any model provider client on a request/config failure."""
