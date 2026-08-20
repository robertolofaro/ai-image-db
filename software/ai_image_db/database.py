"""SQLite database helper for ai-image-db."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


class Database:
    def __init__(self, db_path: str | Path = "ai_images.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        if SCHEMA_PATH.exists():
            sql = SCHEMA_PATH.read_text(encoding="utf-8")
            self.conn.executescript(sql)
        else:
            # Minimal fallback if schema file missing
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL UNIQUE
                )
                """
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def upsert_image(self, data: dict[str, Any]) -> int:
        """Insert or update an image row. Returns image id."""
        filepath = data["filepath"]
        existing = self.conn.execute(
            "SELECT id FROM images WHERE filepath = ?", (filepath,)
        ).fetchone()

        cols = [
            "filename", "filepath", "width", "height", "filesize_bytes",
            "generation_date", "file_mtime", "author", "software",
            "source_tool", "has_workflow", "has_c2pa", "has_watermark",
            "has_signature",
        ]
        values = [data.get(c) for c in cols]

        if existing:
            image_id = existing["id"]
            sets = ", ".join(f"{c} = ?" for c in cols)
            self.conn.execute(
                f"UPDATE images SET {sets}, updated_at = datetime('now') WHERE id = ?",
                values + [image_id],
            )
        else:
            placeholders = ", ".join("?" for _ in cols)
            cur = self.conn.execute(
                f"INSERT INTO images ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
            image_id = cur.lastrowid

        self.conn.commit()
        return int(image_id)

    def get_image_by_path(self, filepath: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM images WHERE filepath = ?", (filepath,)
        ).fetchone()

    def get_image(self, image_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()

    def list_images(
        self,
        source_tool: Optional[str] = None,
        has_workflow: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM images WHERE 1=1"
        params: list[Any] = []
        if source_tool is not None:
            q += " AND source_tool = ?"
            params.append(source_tool)
        if has_workflow is not None:
            q += " AND has_workflow = ?"
            params.append(1 if has_workflow else 0)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return self.conn.execute(q, params).fetchall()

    # ------------------------------------------------------------------
    # Workflows / EXIF / C2PA / Watermarks / Captions
    # ------------------------------------------------------------------
    def set_workflow(self, image_id: int, workflow: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM workflows WHERE image_id = ?", (image_id,))
        self.conn.execute(
            """
            INSERT INTO workflows (image_id, workflow_json, prompt_json, parameters_text, raw_metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                image_id,
                _json_dumps(workflow.get("workflow")),
                _json_dumps(workflow.get("prompt")),
                workflow.get("parameters_text"),
                workflow.get("raw_metadata"),
            ),
        )
        self.conn.execute(
            "UPDATE images SET has_workflow = 1, updated_at = datetime('now') WHERE id = ?",
            (image_id,),
        )
        self.conn.commit()

    def set_exif(self, image_id: int, exif: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM exif_data WHERE image_id = ?", (image_id,))
        for tag, value in exif.items():
            if value is None:
                continue
            self.conn.execute(
                "INSERT OR REPLACE INTO exif_data (image_id, tag, value) VALUES (?, ?, ?)",
                (image_id, str(tag), _to_str(value)),
            )
        self.conn.commit()

    def set_c2pa(self, image_id: int, info: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM c2pa_info WHERE image_id = ?", (image_id,))
        self.conn.execute(
            """
            INSERT INTO c2pa_info (
                image_id, has_manifest, is_valid, active_manifest,
                validation_status, claim_generator, title,
                assertions_summary, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                1 if info.get("has_manifest") else 0,
                info.get("is_valid"),
                _json_dumps(info.get("active_manifest")),
                _json_dumps(info.get("validation_status")),
                info.get("claim_generator"),
                info.get("title"),
                _json_dumps(info.get("assertions_summary")),
                _json_dumps(info.get("raw_json")),
            ),
        )
        self.conn.execute(
            "UPDATE images SET has_c2pa = ?, updated_at = datetime('now') WHERE id = ?",
            (1 if info.get("has_manifest") else 0, image_id),
        )
        self.conn.commit()

    def add_watermark(self, image_id: int, record: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO watermarks (image_id, kind, method, detected, confidence, details)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                record.get("kind", "unknown"),
                record.get("method"),
                1 if record.get("detected") else 0,
                record.get("confidence"),
                _json_dumps(record.get("details")) if not isinstance(record.get("details"), str) else record.get("details"),
            ),
        )
        if record.get("detected"):
            flag = "has_signature" if record.get("kind") in ("signature", "user-signature") else "has_watermark"
            self.conn.execute(
                f"UPDATE images SET {flag} = 1, updated_at = datetime('now') WHERE id = ?",
                (image_id,),
            )
        self.conn.commit()

    def set_caption(self, image_id: int, caption: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM captions WHERE image_id = ?", (image_id,))
        tags = caption.get("tags")
        if isinstance(tags, list):
            tags_str = ", ".join(str(t) for t in tags)
        else:
            tags_str = tags
        self.conn.execute(
            """
            INSERT INTO captions (
                image_id, narrative, short_caption, generated_prompt,
                tags, tags_json, model_florence, model_wd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                caption.get("narrative"),
                caption.get("short_caption"),
                caption.get("generated_prompt"),
                tags_str,
                _json_dumps(caption.get("tags_json")),
                caption.get("model_florence"),
                caption.get("model_wd"),
            ),
        )
        self.conn.commit()

    def get_full_record(self, image_id: int) -> dict[str, Any]:
        img = self.get_image(image_id)
        if not img:
            return {}
        out = dict(img)
        wf = self.conn.execute(
            "SELECT * FROM workflows WHERE image_id = ?", (image_id,)
        ).fetchone()
        out["workflow"] = dict(wf) if wf else None
        out["exif"] = {
            r["tag"]: r["value"]
            for r in self.conn.execute(
                "SELECT tag, value FROM exif_data WHERE image_id = ?", (image_id,)
            )
        }
        c2 = self.conn.execute(
            "SELECT * FROM c2pa_info WHERE image_id = ?", (image_id,)
        ).fetchone()
        out["c2pa"] = dict(c2) if c2 else None
        out["watermarks"] = [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM watermarks WHERE image_id = ?", (image_id,)
            )
        ]
        cap = self.conn.execute(
            "SELECT * FROM captions WHERE image_id = ?", (image_id,)
        ).fetchone()
        out["caption"] = dict(cap) if cap else None
        return out


def _json_dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _to_str(v: Any) -> str:
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)
    return str(v)
