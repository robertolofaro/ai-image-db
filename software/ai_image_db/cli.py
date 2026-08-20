"""
Command-line interface for the ai-image-db.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .database import Database
from .loader import ImageLoader
from .captioner import Captioner
from .provenance import ProvenanceChecker


@click.group()
@click.option("--db", default="ai_images.db", help="SQLite database path")
@click.pass_context
def main(ctx, db):
    """AI Image Provenance – load, caption, and audit AI-generated images."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


@main.command("load")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=False)
@click.option("--no-c2pa", is_flag=True, help="Skip C2PA extraction")
@click.option("--no-wm", is_flag=True, help="Skip watermark heuristics")
@click.option(
    "--error-csv",
    default="load_errors.csv",
    show_default=True,
    help="CSV file for failed loads (path, reason, timestamp). Empty string disables.",
)
@click.pass_context
def load_cmd(ctx, paths, recursive, no_c2pa, no_wm, error_csv):
    """Load one or more images (or directories) into the database.

    Failures are appended to --error-csv with columns: path, reason, timestamp.
    """
    if not paths:
        raise click.UsageError("Provide at least one file or directory")
    err = error_csv if error_csv else None
    with ImageLoader(
        ctx.obj["db_path"],
        check_c2pa=not no_c2pa,
        check_watermarks=not no_wm,
        error_csv=err,
    ) as loader:
        ids = loader.load_many(paths, recursive=recursive)
        click.echo(
            f"Loaded {len(ids)} image(s). IDs: {ids[:20]}{'…' if len(ids) > 20 else ''}"
        )
        if err:
            click.echo(f"Failures (if any) written to: {err}")


@main.command("caption")
@click.argument("image_ids", nargs=-1, type=int)
@click.option("--all", "do_all", is_flag=True, help="Caption every image that has no caption yet")
@click.option("--florence", default="microsoft/Florence-2-base")
@click.option("--wd", default="SmilingWolf/wd-v1-4-vit-tagger-v2")
@click.option("--threshold", default=0.35, type=float)
@click.pass_context
def caption_cmd(ctx, image_ids, do_all, florence, wd, threshold):
    """Run Florence-2 + WD tagger and store results."""
    db = Database(ctx.obj["db_path"])
    cap = Captioner(florence_model=florence, wd_model=wd, wd_threshold=threshold)

    targets = list(image_ids)
    if do_all:
        rows = db.conn.execute(
            """
            SELECT i.id FROM images i
            LEFT JOIN captions c ON c.image_id = i.id
            WHERE c.id IS NULL
            """
        ).fetchall()
        targets = [r["id"] for r in rows]

    if not targets:
        click.echo("No images to caption.")
        return

    for iid in targets:
        click.echo(f"Captioning image id={iid} …")
        try:
            result = cap.caption_and_store(db, iid)
            click.echo(f"  short: {result.get('short_caption', '')[:120]}")
            click.echo(f"  tags : {', '.join(result.get('tags', [])[:12])} …")
        except Exception as e:
            click.echo(f"  ERROR: {e}")
    db.close()


@main.command("audit")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json-out", "json_out", is_flag=True)
@click.pass_context
def audit_cmd(ctx, path, json_out):
    """Check C2PA / watermark / signature on an external image (EU AI Act Art. 50 oriented)."""
    checker = ProvenanceChecker()
    report = checker.audit(path)
    if json_out:
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        art = report["art50_oriented"]
        click.echo(f"File: {report['path']}")
        click.echo(f"  Machine-readable mark present : {art['machine_readable_mark_present']}")
        click.echo(f"  Human-visible label/signature : {art['human_visible_label_or_signature']}")
        c2 = report["c2pa"]
        click.echo(f"  C2PA manifest                 : {c2.get('has_manifest')} (valid={c2.get('is_valid')})")
        if c2.get("claim_generator"):
            click.echo(f"  Claim generator               : {c2.get('claim_generator')}")
        for m in report["watermarks"]:
            if m.get("detected"):
                click.echo(f"  Watermark/signature           : {m.get('kind')} via {m.get('method')}")


@main.command("show")
@click.argument("image_id", type=int)
@click.pass_context
def show_cmd(ctx, image_id):
    """Print full stored record for an image id."""
    db = Database(ctx.obj["db_path"])
    rec = db.get_full_record(image_id)
    db.close()
    if not rec:
        click.echo("Not found")
        return
    # Avoid dumping huge workflow JSON by default
    if rec.get("workflow") and rec["workflow"].get("workflow_json"):
        w = rec["workflow"]
        w = dict(w)
        wj = w.get("workflow_json") or ""
        if len(wj) > 400:
            w["workflow_json"] = wj[:400] + f"… ({len(wj)} chars)"
        rec["workflow"] = w
    click.echo(json.dumps(rec, indent=2, default=str))


@main.command("list")
@click.option("--source", default=None)
@click.option("--limit", default=20, type=int)
@click.pass_context
def list_cmd(ctx, source, limit):
    """List images in the database."""
    db = Database(ctx.obj["db_path"])
    rows = db.list_images(source_tool=source, limit=limit)
    for r in rows:
        click.echo(
            f"{r['id']:4d}  {r['filename'][:40]:40s}  "
            f"{r['width']}x{r['height']}  source={r['source_tool']}  "
            f"wf={r['has_workflow']} c2pa={r['has_c2pa']}"
        )
    db.close()


if __name__ == "__main__":
    main()
