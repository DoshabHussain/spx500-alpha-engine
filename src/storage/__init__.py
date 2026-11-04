"""
Storage layer for SPX500 forecasting system.

Provides:
- Raw data store (append-only, partitioned)
- File manifests for reproducibility
"""

from src.storage.manifests import ManifestManager, get_manifest_manager
from src.storage.raw_store import RawStore, get_raw_store

__all__ = [
    "RawStore",
    "get_raw_store",
    "ManifestManager",
    "get_manifest_manager",
]
