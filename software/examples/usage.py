"""
Minimal end-to-end example.

Usage:
  python examples/usage.py /path/to/image.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_image_db import Database, ImageLoader, Captioner, ProvenanceChecker


def main(image_path: str):
    db_path = "example_ai_images.db"
    path = Path(image_path)

    print("=== 1. Load into SQLite ===")
    with ImageLoader(db_path, check_c2pa=True, check_watermarks=True) as loader:
        image_id = loader.load_one(path)
    print(f"  image_id = {image_id}")

    print("\n=== 2. Full record (before caption) ===")
    db = Database(db_path)
    rec = db.get_full_record(image_id)
    print(f"  size: {rec.get('width')}x{rec.get('height')}, {rec.get('filesize_bytes')} bytes")
    print(f"  source_tool: {rec.get('source_tool')}")
    print(f"  has_workflow: {rec.get('has_workflow')}")
    print(f"  has_c2pa: {rec.get('has_c2pa')}")
    print(f"  author: {rec.get('author')}")

    print("\n=== 3. Provenance audit (Art. 50 oriented) ===")
    checker = ProvenanceChecker()
    report = checker.audit(path)
    art = report["art50_oriented"]
    print(f"  machine_readable_mark_present: {art['machine_readable_mark_present']}")
    print(f"  human_visible_label_or_signature: {art['human_visible_label_or_signature']}")

    # Captioning is optional / heavy – uncomment when models are available
    # print("\n=== 4. Caption (Florence-2 + WD) ===")
    # cap = Captioner()
    # result = cap.caption_and_store(db, image_id)
    # print("  short:", result.get("short_caption"))
    # print("  tags:", result.get("tags")[:15])

    db.close()
    print("\nDone. Database:", db_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/usage.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])
