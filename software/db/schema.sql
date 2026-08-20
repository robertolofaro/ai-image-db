-- AI Image Provenance Database Schema
-- Supports ComfyUI / A1111 / other generators + captioning + C2PA / watermark checks

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS images (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    filename            TEXT NOT NULL,               -- original basename
    filepath            TEXT NOT NULL UNIQUE,         -- absolute or relative path used at ingest
    width               INTEGER,
    height              INTEGER,
    filesize_bytes      INTEGER,
    generation_date     TEXT,                        -- ISO-8601 if available, else file mtime
    file_mtime          TEXT,                        -- filesystem mtime as fallback
    author              TEXT,                        -- from EXIF / XMP / C2PA / custom
    software            TEXT,                        -- e.g. "ComfyUI", "A1111", etc.
    source_tool         TEXT,                        -- normalized: comfyui, a1111, midjourney, unknown, external
    has_workflow        INTEGER DEFAULT 0,           -- boolean
    has_c2pa            INTEGER DEFAULT 0,
    has_watermark       INTEGER DEFAULT 0,
    has_signature       INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (datetime('now')),
    updated_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workflows (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL UNIQUE REFERENCES images(id) ON DELETE CASCADE,
    workflow_json       TEXT,                        -- full ComfyUI workflow graph (pretty or raw)
    prompt_json         TEXT,                        -- ComfyUI API-style prompt dict
    parameters_text     TEXT,                        -- A1111-style parameters string if present
    raw_metadata        TEXT,                        -- any other raw text chunks
    extracted_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exif_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tag                 TEXT NOT NULL,
    value               TEXT,
    UNIQUE(image_id, tag)
);

CREATE TABLE IF NOT EXISTS c2pa_info (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL UNIQUE REFERENCES images(id) ON DELETE CASCADE,
    has_manifest        INTEGER DEFAULT 0,
    is_valid            INTEGER,                     -- null = not checked / error
    active_manifest     TEXT,                        -- JSON of active manifest summary
    validation_status   TEXT,                        -- JSON array or text
    claim_generator     TEXT,
    title               TEXT,
    assertions_summary  TEXT,                        -- JSON
    raw_json            TEXT,                        -- full reader.json() dump
    checked_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watermarks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    kind                TEXT NOT NULL,               -- 'visible', 'invisible', 'signature', 'c2pa', 'custom', 'unknown'
    method              TEXT,                        -- e.g. 'invisible-watermark', 'dwt', 'user-signature', 'exif'
    detected            INTEGER DEFAULT 0,
    confidence          REAL,
    details             TEXT,                        -- free-form / JSON
    checked_at          TEXT DEFAULT (datetime('now'))
);

-- Captioning results (Florence-2 + WD14-style tags)
CREATE TABLE IF NOT EXISTS captions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL UNIQUE REFERENCES images(id) ON DELETE CASCADE,
    narrative           TEXT,                        -- long descriptive caption (Florence-2 MORE_DETAILED)
    short_caption       TEXT,                        -- Florence-2 CAPTION
    generated_prompt    TEXT,                        -- prompt-style reconstruction from image
    tags                TEXT,                        -- comma-separated or JSON list of WD tags
    tags_json           TEXT,                        -- structured [{"tag": "...", "score": 0.xx}, ...]
    model_florence      TEXT,                        -- model id used
    model_wd            TEXT,
    captioned_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath);
CREATE INDEX IF NOT EXISTS idx_images_source ON images(source_tool);
CREATE INDEX IF NOT EXISTS idx_images_has_workflow ON images(has_workflow);
CREATE INDEX IF NOT EXISTS idx_exif_image ON exif_data(image_id);
CREATE INDEX IF NOT EXISTS idx_watermarks_image ON watermarks(image_id);
