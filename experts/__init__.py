"""Capability routing for Veya Core and external NIM experts.

This package does not import the Veya checkpoint and does not call a provider.
"""

from experts.router import route

__all__ = ["route"]
