"""Phase 0, step 4: download GraphCast_small, its normalisation stats and the
1-degree ERA5 example files from the authors' public bucket (no login needed).

    python scripts/00_download_assets.py
"""
from extremes import model, paths

if __name__ == "__main__":
    dest = model.download_assets()
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(paths.DATA_DIR)}  {p.stat().st_size / 1e6:.1f} MB")
