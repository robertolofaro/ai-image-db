"""
Captioning module using Florence-2 (narrative / prompt) and WD14-style tagger.

Results are written into the `captions` table – derived from the *image*,
not from any embedded original prompt.

Supports:
  - Local model directories (preferred when the path exists on disk)
  - Hugging Face hub IDs as fallback
  - Compatibility patch for Florence-2 + newer transformers
    ('forced_bos_token_id' AttributeError)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


def _is_local_model_dir(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # Florence / HF-style dir usually has config.json; WD often has model.onnx
    return (
        (p / "config.json").exists()
        or (p / "model.onnx").exists()
        or (p / "selected_tags.csv").exists()
        or any(p.glob("*.safetensors"))
        or any(p.glob("*.bin"))
    )


def _patch_florence_config(model) -> None:
    """
    Newer transformers expect config.forced_bos_token_id during generate().
    Florence-2 remote code configs sometimes omit it → AttributeError.
    """
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return
        # Top-level
        if not hasattr(cfg, "forced_bos_token_id"):
            bos = getattr(cfg, "bos_token_id", None)
            try:
                cfg.forced_bos_token_id = bos
            except Exception:
                pass
        # Nested text_config (Florence2Config)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and not hasattr(text_cfg, "forced_bos_token_id"):
            bos = getattr(text_cfg, "bos_token_id", None) or getattr(cfg, "bos_token_id", None)
            try:
                text_cfg.forced_bos_token_id = bos
            except Exception:
                pass
    except Exception as e:
        print(f"[Captioner] config patch warning: {e}")


class Captioner:
    def __init__(
        self,
        florence_model: str = "microsoft/Florence-2-base",
        wd_model: str = "SmilingWolf/wd-v1-4-vit-tagger-v2",
        device: Optional[str] = None,
        wd_threshold: float = 0.35,
        local_files_only: Optional[bool] = None,
    ):
        """
        florence_model / wd_model: HF repo id OR absolute/relative local directory.

        local_files_only:
          - True  → never hit the Hub
          - False → always allow Hub
          - None  → auto: True when path is an existing local model dir
        """
        self.florence_model_id = florence_model
        self.wd_model_id = wd_model
        self.wd_threshold = wd_threshold
        self.device = device
        self.local_files_only = local_files_only
        self._florence = None
        self._florence_processor = None
        self._wd_model = None
        self._wd_tags = None
        self._wd_input_name = None

    # ------------------------------------------------------------------
    # Florence-2
    # ------------------------------------------------------------------
    def _load_florence(self):
        if self._florence is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model_id = self.florence_model_id
        is_local = _is_local_model_dir(model_id)
        local_only = (
            self.local_files_only if self.local_files_only is not None else is_local
        )

        print(
            f"[Captioner] Loading Florence-2 ({model_id}) on {device} "
            f"(local_files_only={local_only}) …"
        )

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_only,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        # attn implementation can fail on some builds; leave default

        try:
            self._florence = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs
            ).to(device).eval()
        except Exception as e:
            # Common: transformers version mismatch with remote modeling code
            print(f"[Captioner] Florence load failed: {e}")
            print(
                "[Captioner] Tip: try  pip install 'transformers>=4.41.0,<4.50'  "
                "or use a Florence build whose modeling_*.py matches your transformers."
            )
            raise

        self._florence_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        _patch_florence_config(self._florence)

    def _florence_task(self, image: Image.Image, task: str) -> str:
        self._load_florence()
        import torch

        # Re-patch in case generate path re-reads config
        _patch_florence_config(self._florence)

        inputs = self._florence_processor(
            text=task, images=image, return_tensors="pt"
        )
        # Move tensors to device
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._florence.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        generated_text = self._florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._florence_processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )
        if isinstance(parsed, dict):
            return str(parsed.get(task, next(iter(parsed.values()), "")))
        return str(parsed)

    # ------------------------------------------------------------------
    # WD14 tagger (local dir or HF)
    # ------------------------------------------------------------------
    def _load_wd(self):
        if self._wd_model is not None:
            return
        try:
            self._load_wd_onnx()
        except Exception as e:
            print(f"[Captioner] ONNX WD load failed ({e}); tags will be empty.")
            self._wd_model = "placeholder"
            self._wd_tags = []

    def _resolve_wd_files(self) -> tuple[Path, Path]:
        """Return (model.onnx path, selected_tags.csv path)."""
        root = Path(self.wd_model_id)
        if _is_local_model_dir(root) or root.is_dir():
            model_file = None
            for name in ("model.onnx", "wd.onnx"):
                cand = root / name
                if cand.exists():
                    model_file = cand
                    break
            if model_file is None:
                found = list(root.glob("*.onnx"))
                if found:
                    model_file = found[0]
            tags_file = None
            for name in ("selected_tags.csv", "tags.csv"):
                cand = root / name
                if cand.exists():
                    tags_file = cand
                    break
            if model_file is None or tags_file is None:
                raise FileNotFoundError(
                    f"Local WD dir {root} needs model.onnx and selected_tags.csv "
                    f"(found model={model_file}, tags={tags_file})"
                )
            return model_file, tags_file

        # Hugging Face hub
        from huggingface_hub import hf_hub_download

        model_file = None
        for name in ("model.onnx", "wd.onnx"):
            try:
                model_file = Path(hf_hub_download(self.wd_model_id, name))
                break
            except Exception:
                continue
        if model_file is None:
            model_file = Path(hf_hub_download(self.wd_model_id, "model.onnx"))

        tags_file = None
        for name in ("selected_tags.csv", "tags.csv"):
            try:
                tags_file = Path(hf_hub_download(self.wd_model_id, name))
                break
            except Exception:
                continue
        if tags_file is None:
            raise FileNotFoundError("Could not find selected_tags.csv for WD model")
        return model_file, tags_file

    def _load_wd_onnx(self):
        import onnxruntime as ort
        import pandas as pd

        print(f"[Captioner] Loading WD tagger ({self.wd_model_id}) …")
        model_file, tags_file = self._resolve_wd_files()
        print(f"[Captioner]   onnx={model_file}")
        print(f"[Captioner]   tags={tags_file}")

        providers = []
        try:
            import torch
            if torch.cuda.is_available():
                providers.append("CUDAExecutionProvider")
        except Exception:
            pass
        providers.append("CPUExecutionProvider")

        self._wd_model = ort.InferenceSession(str(model_file), providers=providers)
        self._wd_input_name = self._wd_model.get_inputs()[0].name

        df = pd.read_csv(tags_file)
        if "name" in df.columns:
            self._wd_tags = df["name"].tolist()
        else:
            self._wd_tags = df.iloc[:, 1].tolist()

    def _wd_tag(self, image: Image.Image) -> list[dict[str, Any]]:
        self._load_wd()
        if self._wd_model == "placeholder":
            return []

        img = image.convert("RGB").resize((448, 448), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB → BGR
        arr = np.expand_dims(arr, 0)

        probs = self._wd_model.run(None, {self._wd_input_name: arr})[0][0]
        results = []
        for tag, score in zip(self._wd_tags, probs):
            if float(score) >= self.wd_threshold:
                results.append({"tag": str(tag), "score": float(score)})
        results.sort(key=lambda x: -x["score"])
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def caption_image(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        image = Image.open(path).convert("RGB")

        narrative = self._florence_task(image, "<MORE_DETAILED_CAPTION>")
        short = self._florence_task(image, "<CAPTION>")
        try:
            prompt_style = self._florence_task(image, "<DETAILED_CAPTION>")
        except Exception:
            prompt_style = narrative

        tags_scored = self._wd_tag(image)
        tags = [t["tag"] for t in tags_scored]

        return {
            "narrative": narrative.strip(),
            "short_caption": short.strip(),
            "generated_prompt": prompt_style.strip(),
            "tags": tags,
            "tags_json": tags_scored,
            "model_florence": self.florence_model_id,
            "model_wd": self.wd_model_id,
        }

    def caption_and_store(
        self,
        db,
        image_id: int,
        path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        if path is None:
            row = db.get_image(image_id)
            if not row:
                raise ValueError(f"No image with id={image_id}")
            path = row["filepath"]
        result = self.caption_image(path)
        db.set_caption(image_id, result)
        return result
