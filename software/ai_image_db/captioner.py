"""
Captioning module using Florence-2 (narrative / prompt) and WD14-style tagger.

Results are written into the `captions` table – derived from the *image*,
not from any embedded original prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# Optional heavy imports are done lazily so the rest of the toolkit still works
# without GPU / torch installed.


class Captioner:
    def __init__(
        self,
        florence_model: str = "microsoft/Florence-2-base",
        wd_model: str = "SmilingWolf/wd-v1-4-vit-tagger-v2",
        device: Optional[str] = None,
        wd_threshold: float = 0.35,
    ):
        self.florence_model_id = florence_model
        self.wd_model_id = wd_model
        self.wd_threshold = wd_threshold
        self.device = device  # "cuda", "cpu", or None (auto)
        self._florence = None
        self._florence_processor = None
        self._wd_model = None
        self._wd_tags = None

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
        print(f"[Captioner] Loading Florence-2 ({self.florence_model_id}) on {device} …")
        self._florence = AutoModelForCausalLM.from_pretrained(
            self.florence_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self._florence_processor = AutoProcessor.from_pretrained(
            self.florence_model_id, trust_remote_code=True
        )

    def _florence_task(self, image: Image.Image, task: str) -> str:
        self._load_florence()
        import torch

        inputs = self._florence_processor(
            text=task, images=image, return_tensors="pt"
        ).to(self.device)
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
        # parsed is usually {task: "caption text"}
        if isinstance(parsed, dict):
            return str(parsed.get(task, next(iter(parsed.values()), "")))
        return str(parsed)

    # ------------------------------------------------------------------
    # WD14 / WD1.4 style tagger (ONNX preferred when available)
    # ------------------------------------------------------------------
    def _load_wd(self):
        if self._wd_model is not None:
            return
        # Prefer ONNX runtime path used by most modern WD taggers
        try:
            self._load_wd_onnx()
        except Exception as e:
            print(f"[Captioner] ONNX WD load failed ({e}); falling back to a simple placeholder.")
            self._wd_model = "placeholder"
            self._wd_tags = []

    def _load_wd_onnx(self):
        """
        Load a WD14-style ONNX model from Hugging Face.
        Common repos: SmilingWolf/wd-v1-4-vit-tagger-v2, wd-vit-tagger-v3, etc.
        """
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        import pandas as pd

        print(f"[Captioner] Loading WD tagger ({self.wd_model_id}) …")
        # Try common file names
        model_file = None
        for name in ("model.onnx", "wd.onnx", "model.onnx"):
            try:
                model_file = hf_hub_download(self.wd_model_id, name)
                break
            except Exception:
                continue
        if model_file is None:
            # some repos ship under different structure
            model_file = hf_hub_download(self.wd_model_id, "model.onnx")

        tags_file = None
        for name in ("selected_tags.csv", "tags.csv"):
            try:
                tags_file = hf_hub_download(self.wd_model_id, name)
                break
            except Exception:
                continue
        if tags_file is None:
            raise FileNotFoundError("Could not find selected_tags.csv for WD model")

        self._wd_model = ort.InferenceSession(
            model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        df = pd.read_csv(tags_file)
        # Standard WD CSV has columns: tag_id, name, category, count
        if "name" in df.columns:
            self._wd_tags = df["name"].tolist()
        else:
            self._wd_tags = df.iloc[:, 1].tolist()

    def _wd_tag(self, image: Image.Image) -> list[dict[str, Any]]:
        self._load_wd()
        if self._wd_model == "placeholder":
            return []

        # Preprocess to 448x448 (common for WD14)
        img = image.convert("RGB").resize((448, 448), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32)
        # WD models usually expect BGR and [0,255] or normalized; most ONNX
        # export expect NHWC float32 in 0-255 BGR.
        arr = arr[:, :, ::-1]  # RGB -> BGR
        arr = np.expand_dims(arr, 0)

        input_name = self._wd_model.get_inputs()[0].name
        probs = self._wd_model.run(None, {input_name: arr})[0][0]

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
        # Prompt-style: ask Florence for a generation-oriented description
        try:
            prompt_style = self._florence_task(
                image, "<DETAILED_CAPTION>"
            )  # slightly different prompt
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
        """
        Run captioning and write results into the captions table.
        If path is omitted, look up filepath from the DB.
        """
        if path is None:
            row = db.get_image(image_id)
            if not row:
                raise ValueError(f"No image with id={image_id}")
            path = row["filepath"]
        result = self.caption_image(path)
        db.set_caption(image_id, result)
        return result
