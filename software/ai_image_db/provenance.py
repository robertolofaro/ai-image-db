"""
Provenance / transparency checker oriented toward EU AI Act Art. 50.

Art. 50 requires (among other things) that providers of generative AI systems
mark outputs in a *machine-readable* format so they are detectable as
artificially generated or manipulated. Common technical approaches include:

  - C2PA Content Credentials (cryptographic manifests)
  - Imperceptible / robust watermarks
  - Signed metadata / signature tokens in EXIF
  - Visible signatures / logos (weaker, but still useful)

This module:
  1. Reads & validates C2PA manifests (via c2pa-python when available)
  2. Looks for EXIF Signature tokens, Artist UUIDs, AI disclosure PNG keys
  3. Returns structured results suitable for storage and compliance review
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS


class ProvenanceChecker:
    def __init__(self):
        self._c2pa_available = False
        try:
            import c2pa  # noqa: F401
            self._c2pa_available = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # C2PA
    # ------------------------------------------------------------------
    def check_c2pa(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        result: dict[str, Any] = {
            "has_manifest": False,
            "is_valid": None,
            "active_manifest": None,
            "validation_status": None,
            "claim_generator": None,
            "title": None,
            "assertions_summary": None,
            "signature_info": None,
            "digital_source_type": None,
            "raw_json": None,
            "error": None,
        }
        if not self._c2pa_available:
            result["error"] = "c2pa-python not installed (pip install c2pa-python)"
            return result

        try:
            import c2pa

            reader = None
            try:
                reader = c2pa.Reader(str(path))
            except TypeError:
                mime = _guess_mime(path)
                with open(path, "rb") as f:
                    reader = c2pa.Reader(mime, f)

            raw = None
            try:
                raw = json.loads(reader.json())
            except Exception:
                try:
                    raw = reader.json()
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                except Exception as e:
                    result["error"] = f"Could not parse C2PA JSON: {e}"
                    return result

            result["raw_json"] = raw
            if not raw:
                return result

            active_key = raw.get("active_manifest")
            manifests = raw.get("manifests") or {}
            active = None
            if active_key and active_key in manifests:
                active = manifests[active_key]
            elif manifests:
                active = next(iter(manifests.values()))

            if active:
                result["has_manifest"] = True
                result["active_manifest"] = active
                cgi = active.get("claim_generator_info") or active.get("claim_generator")
                if isinstance(cgi, list) and cgi:
                    result["claim_generator"] = cgi[0].get("name") or str(cgi[0])
                else:
                    result["claim_generator"] = cgi
                result["title"] = active.get("title")
                result["signature_info"] = active.get("signature_info")

                assertions = active.get("assertions") or []
                summary = []
                for a in assertions:
                    entry = {
                        "label": a.get("label"),
                        "data_keys": list((a.get("data") or {}).keys())
                        if isinstance(a.get("data"), dict)
                        else None,
                    }
                    data = a.get("data") or {}
                    if a.get("label") in ("c2pa.actions", "c2pa.actions.v2"):
                        for act in data.get("actions") or []:
                            if act.get("digitalSourceType"):
                                result["digital_source_type"] = act["digitalSourceType"]
                            if act.get("softwareAgent") and not result["claim_generator"]:
                                result["claim_generator"] = act["softwareAgent"]
                    if a.get("label") == "c2pa.creative_work":
                        authors = data.get("author") or []
                        if authors:
                            entry["authors"] = authors
                    summary.append(entry)
                result["assertions_summary"] = summary

            vs = raw.get("validation_status") or raw.get("validationStatus")
            result["validation_status"] = vs
            if vs is not None and isinstance(vs, list):
                result["is_valid"] = not any(
                    "invalid" in str(x.get("code", "")).lower()
                    or "failure" in str(x.get("code", "")).lower()
                    for x in vs
                )
            else:
                result["is_valid"] = result["has_manifest"]

        except Exception as e:
            msg = str(e).lower()
            if any(
                x in msg
                for x in (
                    "no manifest",
                    "not found",
                    "no jumbf",
                    "missing",
                    "no c2pa",
                    "manifestnotfound",
                )
            ):
                result["has_manifest"] = False
            else:
                result["error"] = str(e)
        return result

    # ------------------------------------------------------------------
    # Watermarks / signatures
    # ------------------------------------------------------------------
    def check_watermarks(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        findings: list[dict[str, Any]] = []

        c2 = self.check_c2pa(path)
        if c2.get("has_manifest"):
            findings.append(
                {
                    "kind": "c2pa",
                    "method": "content-credentials",
                    "detected": True,
                    "confidence": 1.0 if c2.get("is_valid") else 0.75,
                    "details": {
                        "claim_generator": c2.get("claim_generator"),
                        "title": c2.get("title"),
                        "is_valid": c2.get("is_valid"),
                        "digital_source_type": c2.get("digital_source_type"),
                        "signature_info": c2.get("signature_info"),
                        "validation_status": c2.get("validation_status"),
                    },
                }
            )

        try:
            with Image.open(path) as img:
                text = getattr(img, "text", None) or {}

                ai_keys = {k: v for k, v in text.items() if k.lower().startswith("ai-")}
                if ai_keys:
                    findings.append(
                        {
                            "kind": "metadata",
                            "method": "png-ai-disclosure",
                            "detected": True,
                            "confidence": 0.95,
                            "details": ai_keys,
                        }
                    )
                if text.get("Description") and "AI" in str(text["Description"]).upper():
                    findings.append(
                        {
                            "kind": "visible",
                            "method": "png-description",
                            "detected": True,
                            "confidence": 0.8,
                            "details": {"Description": str(text["Description"])[:300]},
                        }
                    )

                for key, val in text.items():
                    kl = key.lower()
                    if kl in {k.lower() for k in ai_keys} or key == "Description":
                        continue
                    if any(
                        x in kl
                        for x in (
                            "watermark",
                            "signature",
                            "copyright",
                            "artist",
                            "creator",
                        )
                    ):
                        findings.append(
                            {
                                "kind": "signature" if "sign" in kl else "visible",
                                "method": f"png-text:{key}",
                                "detected": True,
                                "confidence": 0.6,
                                "details": {"key": key, "value": str(val)[:500]},
                            }
                        )

                try:
                    exif = img.getexif()
                    if exif:
                        artist = exif.get(315)
                        desc = exif.get(270)
                        user_comment = None
                        try:
                            ifd = exif.get_ifd(0x8769)
                            user_comment = ifd.get(37510)
                            if isinstance(user_comment, bytes):
                                user_comment = user_comment.decode("utf-8", errors="replace")
                        except Exception:
                            pass

                        if artist:
                            findings.append(
                                {
                                    "kind": "signature",
                                    "method": "exif-artist",
                                    "detected": True,
                                    "confidence": 0.7,
                                    "details": {"Artist": str(artist)},
                                }
                            )

                        for field_name, field_val in (
                            ("ImageDescription", desc),
                            ("UserComment", user_comment),
                        ):
                            if not field_val:
                                continue
                            text_val = str(field_val).replace("ASCII\x00\x00\x00", "").strip()
                            m = re.search(r"Signature:\s*(\S+)", text_val, re.IGNORECASE)
                            if m:
                                token = m.group(1)
                                findings.append(
                                    {
                                        "kind": "signature",
                                        "method": f"exif-{field_name.lower()}",
                                        "detected": True,
                                        "confidence": 0.9,
                                        "details": {
                                            "token_prefix": token[:64]
                                            + ("…" if len(token) > 64 else ""),
                                            "token_length": len(token),
                                        },
                                    }
                                )
                            elif "signature" in text_val.lower() or len(text_val) > 100:
                                findings.append(
                                    {
                                        "kind": "signature",
                                        "method": f"exif-{field_name.lower()}",
                                        "detected": True,
                                        "confidence": 0.5,
                                        "details": {"value_prefix": text_val[:120]},
                                    }
                                )
                except Exception:
                    pass
        except Exception as e:
            findings.append(
                {
                    "kind": "unknown",
                    "method": "error",
                    "detected": False,
                    "confidence": 0.0,
                    "details": {"error": str(e)},
                }
            )

        return findings

    def audit(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        c2 = self.check_c2pa(path)
        marks = self.check_watermarks(path)

        machine_readable = bool(c2.get("has_manifest")) or any(
            m.get("kind") in ("c2pa", "invisible", "metadata") and m.get("detected")
            for m in marks
        )
        human_visible = any(
            m.get("kind") in ("visible", "signature") and m.get("detected") for m in marks
        )
        has_signature = any(
            m.get("kind") == "signature" and m.get("detected") for m in marks
        )

        return {
            "path": str(path),
            "c2pa": {k: v for k, v in c2.items() if k != "raw_json"},
            "watermarks": marks,
            "art50_oriented": {
                "machine_readable_mark_present": machine_readable,
                "human_visible_label_or_signature": human_visible,
                "cryptographic_signature_present": has_signature
                or bool(c2.get("signature_info")),
                "claim_generator": c2.get("claim_generator"),
                "digital_source_type": c2.get("digital_source_type"),
                "notes": (
                    "EU AI Act Art. 50 requires providers to apply effective, "
                    "interoperable, robust, machine-readable marks so AI-generated "
                    "or manipulated content is detectable. C2PA Content Credentials "
                    "and robust watermarks are common approaches. This audit only "
                    "reports what is present; it does not certify legal compliance."
                ),
            },
        }


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
    }.get(ext, "application/octet-stream")
