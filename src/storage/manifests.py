"""
File manifest management for reproducibility.

Tracks exact files used in each pipeline run with:
- File paths
- Cryptographic hashes (SHA-256)
- File sizes
- Row counts
- Creation timestamps

Enables exact reproduction of results.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core import get_config, get_logger
from src.core.schema import FileManifest


logger = get_logger(__name__)


# ─── MANIFEST MANAGER ───────────────────────────────────────────────────────

class ManifestManager:
    """
    Manage file manifests for reproducibility.
    
    Tracks all files used in pipeline runs with cryptographic hashes.
    """
    
    def __init__(self, manifest_dir: Optional[Path] = None) -> None:
        """
        Initialize manifest manager.
        
        Args:
            manifest_dir: Directory for manifest files (default: runs/manifests)
        """
        config = get_config()
        self.manifest_dir = manifest_dir or Path(config.storage.runs_dir) / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ManifestManager initialized", dir=str(self.manifest_dir))
    
    def create_manifest(
        self,
        filepath: Path,
        data_source: Optional[str] = None,
    ) -> FileManifest:
        """
        Create a manifest entry for a file.
        
        Args:
            filepath: Path to file
            data_source: Data source name (e.g., "yfinance", "orats")
        
        Returns:
            FileManifest object
        
        Examples:
            >>> manager = ManifestManager()
            >>> manifest = manager.create_manifest(Path("data/raw/SPY_20250206.parquet"))
            >>> print(manifest.file_hash[:16])
            a1b2c3d4e5f6g7h8
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Calculate hash
        file_hash = self._calculate_sha256(filepath)
        
        # Get file size
        file_size = filepath.stat().st_size
        
        # Try to get row count for parquet files
        row_count = None
        if filepath.suffix == ".parquet":
            try:
                df = pd.read_parquet(filepath)
                row_count = len(df)
            except Exception as e:
                logger.warning("Could not read parquet file", filepath=str(filepath), error=str(e))
        
        manifest = FileManifest(
            filepath=str(filepath),
            file_hash=file_hash,
            file_size=file_size,
            created_at=datetime.now(),
            row_count=row_count,
            data_source=data_source,
        )
        
        logger.debug(
            "Created manifest",
            filepath=str(filepath),
            hash=file_hash[:16],
            size_mb=round(file_size / 1024 / 1024, 2),
        )
        
        return manifest
    
    def save_manifest(
        self,
        run_id: str,
        manifests: List[FileManifest],
        manifest_type: str = "input",
    ) -> Path:
        """
        Save manifest list to JSON file.
        
        Args:
            run_id: Pipeline run ID
            manifests: List of file manifests
            manifest_type: Type (input, output, etc.)
        
        Returns:
            Path to saved manifest file
        
        Examples:
            >>> manager = ManifestManager()
            >>> manifests = [manager.create_manifest(Path("file1.parquet"))]
            >>> path = manager.save_manifest("run_20250206", manifests, "input")
        """
        filename = f"{run_id}_{manifest_type}_manifest.json"
        filepath = self.manifest_dir / filename
        
        # Convert to dict
        manifest_data = {
            "run_id": run_id,
            "manifest_type": manifest_type,
            "created_at": datetime.now().isoformat(),
            "file_count": len(manifests),
            "files": [m.model_dump() for m in manifests],
        }
        
        # Write JSON
        with open(filepath, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)
        
        logger.info(
            "Saved manifest",
            run_id=run_id,
            type=manifest_type,
            files=len(manifests),
            path=str(filepath),
        )
        
        return filepath
    
    def load_manifest(
        self,
        run_id: str,
        manifest_type: str = "input",
    ) -> List[FileManifest]:
        """
        Load manifest from JSON file.
        
        Args:
            run_id: Pipeline run ID
            manifest_type: Type (input, output, etc.)
        
        Returns:
            List of file manifests
        """
        filename = f"{run_id}_{manifest_type}_manifest.json"
        filepath = self.manifest_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Manifest not found: {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        manifests = [FileManifest(**file_data) for file_data in data["files"]]
        
        logger.info(
            "Loaded manifest",
            run_id=run_id,
            type=manifest_type,
            files=len(manifests),
        )
        
        return manifests
    
    def verify_manifest(
        self,
        manifests: List[FileManifest],
    ) -> Dict[str, Any]:
        """
        Verify that files match their manifests.
        
        Args:
            manifests: List of file manifests to verify
        
        Returns:
            Verification results with status and any mismatches
        
        Examples:
            >>> manager = ManifestManager()
            >>> manifests = manager.load_manifest("run_20250206", "input")
            >>> results = manager.verify_manifest(manifests)
            >>> print(results["status"])
            'ok'
        """
        results = {
            "status": "ok",
            "verified_count": 0,
            "missing_count": 0,
            "mismatch_count": 0,
            "missing_files": [],
            "hash_mismatches": [],
        }
        
        for manifest in manifests:
            filepath = Path(manifest.filepath)
            
            # Check if file exists
            if not filepath.exists():
                results["missing_files"].append(str(filepath))
                results["missing_count"] += 1
                results["status"] = "error"
                continue
            
            # Verify hash
            current_hash = self._calculate_sha256(filepath)
            if current_hash != manifest.file_hash:
                results["hash_mismatches"].append({
                    "filepath": str(filepath),
                    "expected": manifest.file_hash,
                    "actual": current_hash,
                })
                results["mismatch_count"] += 1
                results["status"] = "error"
                continue
            
            results["verified_count"] += 1
        
        logger.info(
            "Manifest verification complete",
            status=results["status"],
            verified=results["verified_count"],
            missing=results["missing_count"],
            mismatches=results["mismatch_count"],
        )
        
        return results
    
    def create_dataset_manifest(
        self,
        dataset_name: str,
        files: List[Path],
        data_source: Optional[str] = None,
    ) -> List[FileManifest]:
        """
        Create manifests for a list of files (e.g., all files in a dataset).
        
        Args:
            dataset_name: Name of the dataset
            files: List of file paths
            data_source: Data source name
        
        Returns:
            List of manifests
        """
        manifests = []
        
        for filepath in files:
            try:
                manifest = self.create_manifest(filepath, data_source)
                manifests.append(manifest)
            except Exception as e:
                logger.error(
                    "Failed to create manifest",
                    filepath=str(filepath),
                    error=str(e),
                )
        
        logger.info(
            "Created dataset manifest",
            dataset=dataset_name,
            files=len(manifests),
        )
        
        return manifests
    
    # ─── INTERNAL HELPERS ───────────────────────────────────────────────────
    
    def _calculate_sha256(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

_default_manager: Optional[ManifestManager] = None


def get_manifest_manager() -> ManifestManager:
    """Get singleton manifest manager instance."""
    global _default_manager
    
    if _default_manager is None:
        _default_manager = ManifestManager()
    
    return _default_manager
