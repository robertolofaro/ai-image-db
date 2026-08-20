# ai-image-db v2.0

## Purpose:

A Python tool to document within an SQLite database images and associated metadata, including watermarks and C2PA- can be extended with custom schemes

## Development process

I did the design + architecture + tests evolving concepts from two tools that created and used before (documentation of images, watermarking with visible marker+watermark check).

Grok received each iteration of prompts and feed-back on execution/adjustments to introduce, to build the different iterations.

## Features

1. **Ingest** AI-generated images (ComfyUI, A1111/Forge, etc.) into a SQLite database, extracting:
   - filename, width/height, filesize
   - generation date (EXIF or filesystem mtime)
   - embedded **workflow** / **prompt** (ComfyUI) or **parameters** (A1111)
   - remaining EXIF/XMP that is *not* part of the workflow
   - author / software
   - C2PA Content Credentials (if present)
   - watermark / signature heuristics

2. **Caption** images with **Florence-2** (narrative + prompt-style text) and a **WD14/WD1.4-style** tagger. Captions are derived from the *pixels*, not from any original embedded prompt, and stored in the same database.

3. **Audit** external images for machine-readable marks (C2PA, common watermark markers) in an **EU AI Act Article 50**-oriented way.

## Install

```bash
cd ai_image_db
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional but recommended:

- **GPU** for Florence-2 / WD tagger (`torch` with CUDA, `onnxruntime-gpu`)
- System `exiftool` if you later extend the extractor

---

## Quick start

### 1. Load images

```bash
python -m ai_image_db.cli --db my_images.db load /path/to/outputs/ --recursive
```

Or a single file:

```bash
python -m ai_image_db.cli --db my_images.db load generation_00042.png
```
If one or more images cannot be processed, the tool produces a CSV file listing:
- filename
- path
- reason 

### 2. Caption (Florence-2 + WD tags)

```bash
# Caption specific IDs
python -m ai_image_db.cli --db my_images.db caption 1 2 3

# Caption everything that still has no caption
python -m ai_image_db.cli --db my_images.db caption --all

# using local models instead of huggingface:
python -m ai_image_db.cli --db <DATABASEPATH>.db caption --all \
  --florence <MODELPATH>/Florence-2-base \
  --wd <MODELPATH>/wd-v1-4-vit-tagger-v2 \
  --local-only

If you receive and error <code>forced_bos_token_id</code>, it is an incompatibility with the latest transformer versions: <code>pip install 'transformers>=4.41.0,<4.50'</code>
```

### 3. Audit an external image (Art. 50 oriented)

```bash
python -m ai_image_db.cli audit /path/to/received_image.png
python -m ai_image_db.cli audit /path/to/received_image.png --json-out
```

### 4. Inspect stored data

```bash
python -m ai_image_db.cli --db my_images.db list
python -m ai_image_db.cli --db my_images.db show 1
```

---

## Python API

```python
from ai_image_db import Database, ImageLoader, Captioner, ProvenanceChecker

db = Database("my_images.db")

# Load
with ImageLoader(db) as loader:
    image_id = loader.load_one("comfy_output.png")

# Caption
cap = Captioner(
    florence_model="microsoft/Florence-2-base",   # or -large
    wd_model="SmilingWolf/wd-v1-4-vit-tagger-v2",
)
result = cap.caption_and_store(db, image_id)
print(result["narrative"])
print(result["tags"])

# Audit external image
checker = ProvenanceChecker()
report = checker.audit("someone_elses_image.jpg")
print(report["art50_oriented"])

# Full record
print(db.get_full_record(image_id))
```

---

## Database schema (summary)

| Table        | Purpose |
|-------------|---------|
| `images`    | Core file info, flags (`has_workflow`, `has_c2pa`, …) |
| `workflows` | ComfyUI workflow / prompt JSON, A1111 parameters |
| `exif_data` | Non-workflow EXIF/XMP key-value pairs |
| `c2pa_info` | C2PA manifest summary + validation |
| `watermarks`| Detected marks / signatures |
| `captions`  | Florence narrative, short caption, generated prompt, WD tags |

See `db/schema.sql` for the full definition.

---

## EU AI Act Article 50 notes

Article 50 requires providers of generative AI systems to ensure outputs are marked in a **machine-readable** format and are **detectable** as AI-generated or manipulated (effectiveness, interoperability, robustness, reliability). Deployers have additional labelling duties for deepfakes and certain public-interest text.

This toolkit:

- Extracts and validates **C2PA Content Credentials** (a widely used cryptographic provenance standard).
- Records other common metadata / watermark markers.
- Produces an `art50_oriented` summary on audit.

It does **not** replace a formal compliance assessment. Proprietary watermarks (e.g. Google SynthID) need vendor detectors. Plug your own detector into `ProvenanceChecker.check_watermarks`if you have one

The tool can be extended- look at **provenance.py** 

**This version does not include the OCR on images (i.e. overlay of a label stating if AI-Generated, AI-Modified, or generically AI) that use locally for test, as it is too resource-intensive and only few images have anyway a visible marker**

If you need to check also for the visible logo, either extend the tool, or use caption and then process captions to spot where a visible AI-logo was added (faster than doing OCR on every single image)

---

## License

Apache2.

Model weights (Florence-2, WD taggers, c2pa-python) remain under their respective licenses.
