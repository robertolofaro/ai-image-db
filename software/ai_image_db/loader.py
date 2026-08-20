"""
High-level loader: extract metadata + optional C2PA/watermark checks and store in DB.

Failed loads are recorded to a CSV file (path + reason).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from .database import Database
from .metadata_extractor import MetadataExtractor
from .provenance import ProvenanceChecker


class ImageLoader:
    def __init__(
        self,
        db: Database | str | Path = "ai_images.db",
        check_c2pa: bool = True,
        check_watermarks: bool = True,
        error_csv: str | Path | None = "load_errors.csv",
    ):
        if isinstance(db, (str, Path)):
            self.db = Database(db)
            self._owns_db = True
        else:
            self.db = db
            self._owns_db = False
        self.extractor = MetadataExtractor()
        self.provenance = ProvenanceChecker() if (check_c2pa or check_watermarks) else None
        self.check_c2pa = check_c2pa
        self.check_watermarks = check_watermarks
        self.error_csv = Path(error_csv) if error_csv else None
        self.errors: list[dict[str, str]] = []

    def close(self) -> None:
        self._flush_errors()
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _record_error(self, path: str | Path, reason: str) -> None:
        entry = {
            "path": str(path),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.errors.append(entry)
        print(f"[ERROR] {path}: {reason}")

    def _flush_errors(self) -> None:
        if not self.error_csv or not self.errors:
            return
        write_header = not self.error_csv.exists()
        with open(self.error_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "reason", "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerows(self.errors)
        print(f"[INFO] Wrote {len(self.errors)} error(s) to {self.error_csv}")
        self.errors.clear()

    def load_one(
        self,
        path: str | Path,
        *,
        run_provenance: Optional[bool] = None,
    ) -> Optional[int]:
        """
        Extract metadata from a single image and store it.
        Returns the image_id, or None on failure (error is logged to CSV).
        """
        path = Path(path)
        try:
            meta = self.extractor.extract(path)
        except Exception as e:
            self._record_error(path, f"extract: {type(e).__name__}: {e}")
            return None

        try:
            image_data = {
                "filename": meta["filename"],
                "filepath": meta["filepath"],
                "width": meta["width"],
                "height": meta["height"],
                "filesize_bytes": meta["filesize_bytes"],
                "generation_date": meta["generation_date"],
                "file_mtime": meta["file_mtime"],
                "author": meta["author"],
                "software": meta["software"],
                "source_tool": meta["source_tool"],
                "has_workflow": 1 if meta.get("workflow") else 0,
                "has_c2pa": 0,
                "has_watermark": 0,
                "has_signature": 1 if meta.get("signature_token") else 0,
            }
            image_id = self.db.upsert_image(image_data)

            if meta.get("workflow"):
                self.db.set_workflow(image_id, meta["workflow"])

            # Store EXIF + AI disclosure + signature token as extra tags
            exif_store = dict(meta.get("exif") or {})
            if meta.get("ai_disclosure"):
                for k, v in meta["ai_disclosure"].items():
                    if k == "_parsed":
                        continue
                    exif_store[f"ai_disclosure.{k}"] = v
            if meta.get("signature_token"):
                exif_store["signature_token"] = meta["signature_token"]
            if exif_store:
                self.db.set_exif(image_id, exif_store)

            # Record signature as a watermark row when present
            if meta.get("signature_token"):
                self.db.add_watermark(
                    image_id,
                    {
                        "kind": "signature",
                        "method": "exif-image-description",
                        "detected": True,
                        "confidence": 0.9,
                        "details": {
                            "token_prefix": meta["signature_token"][:64] + "…",
                            "token_length": len(meta["signature_token"]),
                            "author": meta.get("author"),
                        },
                    },
                )

            do_prov = self.check_c2pa or self.check_watermarks
            if run_provenance is not None:
                do_prov = run_provenance
            if do_prov and self.provenance is not None:
                self._run_provenance(image_id, path)
                # Refine source_tool from C2PA claim generator when still unknown
                row = self.db.get_image(image_id)
                if row and row["source_tool"] in ("unknown", "other"):
                    c2 = self.db.conn.execute(
                        "SELECT claim_generator FROM c2pa_info WHERE image_id = ?",
                        (image_id,),
                    ).fetchone()
                    if c2 and c2["claim_generator"]:
                        cg = str(c2["claim_generator"]).lower()
                        tool = "other"
                        if "grok" in cg or "imagine" in cg:
                            tool = "grok_imagine"
                        elif "comfy" in cg:
                            tool = "comfyui"
                        self.db.conn.execute(
                            "UPDATE images SET source_tool = ?, software = COALESCE(software, ?), updated_at = datetime('now') WHERE id = ?",
                            (tool, c2["claim_generator"], image_id),
                        )
                        self.db.conn.commit()

            return image_id
        except Exception as e:
            self._record_error(path, f"store: {type(e).__name__}: {e}")
            return None

    def load_many(
        self,
        paths: Iterable[str | Path],
        *,
        recursive: bool = False,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"),
        show_progress: bool = True,
    ) -> list[int]:
        files: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                if recursive:
                    for ext in extensions:
                        files.extend(p.rglob(f"*{ext}"))
                        files.extend(p.rglob(f"*{ext.upper()}"))
                else:
                    for ext in extensions:
                        files.extend(p.glob(f"*{ext}"))
                        files.extend(p.glob(f"*{ext.upper()}"))
        seen = set()
        unique = []
        for f in files:
            r = str(f.resolve())
            if r not in seen:
                seen.add(r)
                unique.append(f)

        ids: list[int] = []
        iterator = tqdm(unique, desc="Loading images") if show_progress else unique
        for f in iterator:
            iid = self.load_one(f)
            if iid is not None:
                ids.append(iid)

        self._flush_errors()
        return ids

    def _run_provenance(self, image_id: int, path: Path) -> None:
        assert self.provenance is not None
        try:
            if self.check_c2pa:
                c2 = self.provenance.check_c2pa(path)
                self.db.set_c2pa(image_id, c2)
            if self.check_watermarks:
                marks = self.provenance.check_watermarks(path)
                for m in marks:
                    # Avoid duplicate signature row if we already stored EXIF signature
                    if m.get("kind") == "signature" and m.get("method", "").startswith("exif"):
                        continue
                    self.db.add_watermark(image_id, m)
        except Exception as e:
            self._record_error(path, f"provenance: {type(e).__name__}: {e}")
