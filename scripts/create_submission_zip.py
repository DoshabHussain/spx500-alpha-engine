import os
import shutil
import zipfile
from pathlib import Path

def create_zip():
    base_dir = Path(__file__).resolve().parent.parent
    output_zip_local = base_dir / "spx500_final_submission.zip"
    artifact_zip = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\cfd7a14d-7621-4253-ba0a-ec22b5a8d72e\spx500_final_submission.zip")
    
    ignore_dirs = {
        ".git", ".pytest_cache", "__pycache__", ".venv", "venv", 
        "node_modules", ".gemini", "day-trader-skill"
    }
    ignore_extensions = {".pyc", ".pyd", ".zip"}

    print(f"Packaging project from: {base_dir}")
    
    # Remove existing zip if present
    if output_zip_local.exists():
        output_zip_local.unlink()

    with zipfile.ZipFile(output_zip_local, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ignore_extensions:
                    continue
                # compute relative path
                rel_path = file_path.relative_to(base_dir)
                zipf.write(file_path, arcname=str(rel_path))

    size_mb = output_zip_local.stat().st_size / (1024 * 1024)
    print(f"Created local archive: {output_zip_local} ({size_mb:.2f} MB)")

    # Copy to artifact directory
    if artifact_zip.parent.exists():
        shutil.copy2(output_zip_local, artifact_zip)
        print(f"Copied to artifact directory: {artifact_zip}")

if __name__ == "__main__":
    create_zip()

