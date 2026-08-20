"""
Extract generation metadata from AI images (ComfyUI, A1111, generic EXIF/XMP).

ComfyUI stores:
  - PNG tEXt chunks: "prompt", "workflow"
  - Sometimes parameters in other keys

A1111 / Forge often use:
  - "parameters" text chunk
  - EXIF UserComment

Also captures:
  - EU AI Act style PNG text keys (ai-generated, ai-regulation, …)
  - EXIF Artist / ImageDescription "Signature: …" tokens
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

try:
    import piexif
except ImportError:
    piexif = None  # type: ignore


class MetadataExtractor:
    """Extract dimensions, dates, workflow, EXIF, author, software, signatures from an image file."""

    # Keys that belong to "workflow / generation recipe" and should not be
    # duplicated into the generic exif_data table.
    WORKFLOW_KEYS = {
        "prompt",
        "workflow",
        "parameters",
        "sd-metadata",
        "Comment",
        "UserComment",
    }

    # Explicit AI transparency / disclosure keys (EU AI Act Art. 50 style)
    AI_DISCLOSURE_KEYS = {
        "ai-generated",
        "ai-regulation",
        "ai-article",
        "ai-system",
        "ai-provider",
        "ai-date",
        "ai-content-type",
        "ai-disclosure",
        "ai-digital-source",
        "ai-metadata-json",
        "Description",
    }

    def extract(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        result: dict[str, Any] = {
            "filename": path.name,
            "filepath": str(path),
            "width": None,
            "height": None,
            "filesize_bytes": path.stat().st_size,
            "generation_date": None,
            "file_mtime": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "author": None,
            "software": None,
            "source_tool": "unknown",
            "workflow": None,
            "exif": {},
            "raw_png_text": {},
            "ai_disclosure": {},       # structured Art.50-style fields
            "signature_token": None,   # e.g. EXIF ImageDescription "Signature: …"
        }

        with Image.open(path) as img:
            result["width"], result["height"] = img.size
            result["software"] = self._guess_software(img)
            result["source_tool"] = self._normalize_source(result["software"], img)

            # PNG text chunks (ComfyUI / A1111 / custom AI labels)
            if hasattr(img, "text") and img.text:
                result["raw_png_text"] = dict(img.text)
                wf = self._parse_comfy_or_a1111(img.text)
                if wf:
                    result["workflow"] = wf
                # EU AI Act / custom disclosure keys
                result["ai_disclosure"] = self._parse_ai_disclosure(img.text)
                if result["ai_disclosure"] and result["source_tool"] in ("unknown", "other"):
                    result["source_tool"] = "ai_labeled"

            # EXIF
            exif_dict = self._extract_exif(img)
            for k, v in list(exif_dict.items()):
                kl = str(k).lower()
                if k in self.WORKFLOW_KEYS or kl in {x.lower() for x in self.WORKFLOW_KEYS}:
                    if result["workflow"] is None and kl in ("parameters", "usercomment", "comment"):
                        result["workflow"] = {
                            "parameters_text": str(v),
                            "workflow": None,
                            "prompt": None,
                            "raw_metadata": str(v),
                        }
                    # Still keep UserComment / ImageDescription for signature parsing
                    if kl in ("usercomment", "imagedescription", "image description"):
                        result["exif"][k] = v
                    continue
                result["exif"][k] = v

            # Author / Artist
            for key in ("Artist", "Author", "Creator", "Copyright", "XPAuthor"):
                if key in result["exif"] and result["exif"][key]:
                    result["author"] = str(result["exif"][key]).strip()
                    break
            if not result["author"]:
                for key in ("author", "ai-provider", "Artist"):
                    if key in (result.get("raw_png_text") or {}):
                        result["author"] = str(result["raw_png_text"][key]).strip()
                        break

            # Signature token from ImageDescription / UserComment
            result["signature_token"] = self._extract_signature_token(result["exif"])

            # Generation date
            result["generation_date"] = self._pick_generation_date(
                result["exif"],
                result.get("ai_disclosure") or {},
                result["file_mtime"],
            )

            # Improve software / source when AI disclosure present
            if result["ai_disclosure"].get("ai-system"):
                if not result["software"]:
                    result["software"] = str(result["ai_disclosure"]["ai-system"])
            if result["ai_disclosure"].get("ai-provider") and not result["author"]:
                result["author"] = str(result["ai_disclosure"]["ai-provider"])

        return result

    # ------------------------------------------------------------------
    def _parse_ai_disclosure(self, text_chunks: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in text_chunks.items():
            if k in self.AI_DISCLOSURE_KEYS or k.lower().startswith("ai-"):
                out[k] = v
        # Prefer structured JSON if present
        if "ai-metadata-json" in out:
            parsed = self._safe_json(out["ai-metadata-json"])
            if isinstance(parsed, dict):
                out["_parsed"] = parsed
        return out

    def _extract_signature_token(self, exif: dict) -> Optional[str]:
        """
        Many online generators (e.g. Grok Imagine) put a long token in
        ImageDescription or UserComment as:
            Signature: <base64-like string>
        """
        candidates = []
        for key in ("ImageDescription", "UserComment", "Image Description", "Comment"):
            if key in exif and exif[key]:
                candidates.append(str(exif[key]))
        for text in candidates:
            # Normalize possible bytes leftovers
            text = text.replace("ASCII\x00\x00\x00", "").strip()
            m = re.search(r"Signature:\s*(\S+)", text, re.IGNORECASE)
            if m:
                return m.group(1)
            # Sometimes the whole field is the token
            if len(text) > 80 and re.match(r"^[A-Za-z0-9+/=_-]+$", text.strip()):
                return text.strip()
        return None

    def _parse_comfy_or_a1111(self, text_chunks: dict) -> Optional[dict[str, Any]]:
        out: dict[str, Any] = {
            "workflow": None,
            "prompt": None,
            "parameters_text": None,
            "raw_metadata": None,
        }
        found = False

        if "workflow" in text_chunks:
            out["workflow"] = self._safe_json(text_chunks["workflow"])
            found = True
        if "prompt" in text_chunks:
            out["prompt"] = self._safe_json(text_chunks["prompt"])
            found = True
        if "parameters" in text_chunks:
            out["parameters_text"] = text_chunks["parameters"]
            found = True

        for alt in ("Comment", "Description", "sd-metadata"):
            if alt in text_chunks and not found:
                val = text_chunks[alt]
                # Skip pure AI disclosure descriptions
                if isinstance(val, str) and val.strip().upper().startswith("AI-GENERATED"):
                    continue
                parsed = self._safe_json(val)
                if isinstance(parsed, dict) and ("workflow" in parsed or "prompt" in parsed):
                    out["workflow"] = parsed.get("workflow")
                    out["prompt"] = parsed.get("prompt")
                    found = True
                else:
                    out["parameters_text"] = str(val)
                    found = True

        if found:
            out["raw_metadata"] = json.dumps(
                {
                    k: (v[:500] + "…" if isinstance(v, str) and len(v) > 500 else v)
                    for k, v in text_chunks.items()
                    if k in ("prompt", "workflow", "parameters")
                },
                ensure_ascii=False,
                default=str,
            )
            return out
        return None

    def _extract_exif(self, img: Image.Image) -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            raw = img.getexif()
            if not raw:
                return data
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, str(tag_id))
                if tag == "GPSInfo" and isinstance(value, dict):
                    gps = {}
                    for gk, gv in value.items():
                        gps[GPSTAGS.get(gk, gk)] = gv
                    data["GPSInfo"] = gps
                else:
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="replace")
                        except Exception:
                            value = repr(value)
                    data[tag] = value

            # Also pull Exif IFD UserComment etc.
            try:
                ifd = raw.get_ifd(0x8769)
                for tid, val in ifd.items():
                    tag = TAGS.get(tid, str(tid))
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8", errors="replace")
                        except Exception:
                            val = repr(val)
                    if tag not in data:
                        data[tag] = val
            except Exception:
                pass
        except Exception:
            pass

        if piexif is not None:
            try:
                if "exif" in img.info:
                    exif_dict = piexif.load(img.info["exif"])
                    for ifd_name in ("0th", "Exif", "GPS", "1st"):
                        ifd = exif_dict.get(ifd_name) or {}
                        for tag, value in ifd.items():
                            name = (
                                piexif.TAGS.get(ifd_name, {}).get(tag, {}).get("name")
                                or str(tag)
                            )
                            if name not in data:
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode("utf-8", errors="replace")
                                    except Exception:
                                        value = repr(value)
                                data[name] = value
            except Exception:
                pass
        return data

    def _guess_software(self, img: Image.Image) -> Optional[str]:
        text = getattr(img, "text", None) or {}
        if "workflow" in text or "prompt" in text:
            return "ComfyUI"
        if "parameters" in text:
            return "Stable Diffusion (A1111/Forge)"
        if any(k.startswith("ai-") for k in text):
            return text.get("ai-system") or "AI-labeled"
        try:
            exif = img.getexif()
            if exif:
                soft = exif.get(305)  # Software
                if soft:
                    return str(soft)
        except Exception:
            pass
        return None

    def _normalize_source(self, software: Optional[str], img: Image.Image) -> str:
        text = getattr(img, "text", None) or {}
        if "workflow" in text or "prompt" in text:
            return "comfyui"
        if "parameters" in text:
            return "a1111"
        if any(k.startswith("ai-") for k in text):
            return "ai_labeled"
        if not software:
            return "unknown"
        s = software.lower()
        if "comfy" in s:
            return "comfyui"
        if "automatic1111" in s or "a1111" in s or "stable diffusion" in s:
            return "a1111"
        if "grok" in s or "imagine" in s:
            return "grok_imagine"
        if "midjourney" in s:
            return "midjourney"
        if "invoke" in s:
            return "invokeai"
        if "novelai" in s:
            return "novelai"
        return "other"

    def _pick_generation_date(
        self, exif: dict, ai_disclosure: dict, fallback: str
    ) -> str:
        if ai_disclosure.get("ai-date"):
            return str(ai_disclosure["ai-date"])
        parsed = ai_disclosure.get("_parsed") or {}
        if parsed.get("generation_date"):
            return str(parsed["generation_date"])
        for key in (
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
            "CreateDate",
            "ModifyDate",
        ):
            if key in exif and exif[key]:
                return str(exif[key])
        return fallback

    @staticmethod
    def _safe_json(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return val
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()
        if not val:
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
