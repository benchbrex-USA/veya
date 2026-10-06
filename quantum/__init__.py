"""Optional quantum-calibration boundary for Veya.

Ising and any other NVIDIA NIM model stay outside the Veya checkpoint.
This package does not import the model runtime.
"""

from quantum.config import load_config
from quantum.service import RequestError, handle_request

__all__ = ["RequestError", "handle_request", "load_config"]
