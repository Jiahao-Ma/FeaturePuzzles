"""FastAPI backend for the promptable-segmentation demo.

Serves the single-page frontend, exposes the prompt/segment/render pipeline over
JSON, and starts a viser viewer per finished run.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import paths, sam_prompt
from .pipeline import PromptSpec, build_prompts
from .segment import DYNAMIC, STATIC
from .session import SessionStore
from .static_ref import ref
from .viser_view import ViserRegistry

app = FastAPI(title="VGGT-Omega Promptable Segmentation Demo")
store = SessionStore()
viser_registry = ViserRegistry(base_port=7861)

UPLOAD_DIR = paths.ensure_runs() / "_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gallery of bundled inputs
# ---------------------------------------------------------------------------
def _gallery() -> List[Dict]:
    items: List[Dict] = []
    root = paths.DYNAMIC_VIDEO_ROOT
    if root.is_dir():
        for group in sorted(p for p in root.iterdir() if p.is_dir()):
            for vid in sorted(group.glob("*.mp4")):
                items.append({
                    "id": str(vid.relative_to(root)),
                    "name": vid.stem.replace("_", " "),
                    "group": group.name.split("_", 1)[-1].replace("_", " "),
                    "kind": "video",
                    "suggested_mode": DYNAMIC,
                })
    if paths.SCANNET_TEST.is_dir():
        for scene in sorted(p for p in paths.SCANNET_TEST.iterdir() if p.is_dir())[:12]:
            items.append({
                "id": f"scannet/{scene.name}",
                "name": scene.name,
                "group": "ScanNet (static, sensor depth)",
                "kind": "scannet",
                "suggested_mode": STATIC,
            })
    return items


def _resolve_source(source_id: str) -> Path:
    if source_id.startswith("scannet/"):
        root, rel = paths.SCANNET_TEST, source_id.split("/", 1)[1]
    elif source_id.startswith("upload/"):
        root, rel = UPLOAD_DIR, source_id.split("/", 1)[1]
    else:
        root, rel = paths.DYNAMIC_VIDEO_ROOT, source_id
    p = (root / rel).resolve()
    if root.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, f"source not found: {source_id}")
    return p


def _jpeg(img_rgb: np.ndarray, quality: int = 88) -> Response:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(500, "jpeg encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((paths.WEB_ROOT / "index.html").read_text())


@app.get("/api/gallery")
def gallery() -> JSONResponse:
    return JSONResponse({"items": _gallery()})


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Accept one video, or several images treated as an ordered multi-view set.

    Each upload gets its own timestamped directory rather than a timestamped
    filename, so the original name survives into the result directory's name.
    """
    bucket = UPLOAD_DIR / str(int(time.time()))
    bucket.mkdir(parents=True, exist_ok=True)

    if len(files) == 1:
        dst = bucket / Path(files[0].filename or "upload").name
        with dst.open("wb") as f:
            shutil.copyfileobj(files[0].file, f)
        return JSONResponse({"id": f"upload/{bucket.name}/{dst.name}",
                             "name": dst.stem, "count": 1})

    ordered = sorted(files, key=lambda u: u.filename or "")
    folder = bucket / f"{Path(ordered[0].filename or 'images').stem}_set"
    folder.mkdir(parents=True, exist_ok=True)
    for i, up in enumerate(ordered):
        dst = folder / f"{i:04d}{Path(up.filename or '').suffix.lower() or '.jpg'}"
        with dst.open("wb") as f:
            shutil.copyfileobj(up.file, f)
    return JSONResponse({"id": f"upload/{bucket.name}/{folder.name}",
                         "name": folder.name, "count": len(files)})


@app.post("/api/session")
def create_session(payload: Dict = Body(...)) -> JSONResponse:
    source = _resolve_source(payload["source"])
    scene_mode = payload.get("scene_mode", DYNAMIC)
    if scene_mode not in (STATIC, DYNAMIC):
        raise HTTPException(400, f"bad scene_mode: {scene_mode}")
    n_frames = int(np.clip(int(payload.get("n_frames", 24)), 4, 96))
    s = store.create(source, scene_mode, n_frames)
    store.submit_prepare(s, long_side=int(payload.get("long_side", 640)))
    s.log(f"[queue] preparing '{source.name}' as {scene_mode} "
          f"({n_frames} frames, {store.jobs.depth()} job(s) ahead)")
    return JSONResponse(s.snapshot())


@app.get("/api/session/{sid}")
def session_status(sid: str) -> JSONResponse:
    try:
        return JSONResponse(store.get(sid).snapshot())
    except KeyError:
        raise HTTPException(404, "unknown session")


@app.get("/api/session/{sid}/frame/{idx}")
def session_frame(sid: str, idx: int, w: int = Query(0)) -> Response:
    s = store.get(sid)
    if s.prep is None:
        raise HTTPException(409, "session not ready")
    img = s.prep.src.images[int(np.clip(idx, 0, s.prep.src.size - 1))]
    if w and w != img.shape[1]:
        h = int(round(img.shape[0] * w / img.shape[1]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return _jpeg(img)


@app.post("/api/session/{sid}/prompt")
def session_prompt(sid: str, payload: Dict = Body(...)) -> Response:
    """Resolve prompts on the reference frame and return an overlay preview.

    Runs inline (SAM on one frame is fast) so clicking feels immediate.
    """
    s = store.get(sid)
    if s.prep is None:
        raise HTTPException(409, "session not ready")
    spec = PromptSpec(
        mode=payload.get("mode", "points"),
        objects=payload.get("objects", []),
        top_k=int(payload.get("top_k", 10)),
        points_per_side=int(payload.get("points_per_side", 32)),
        ref_view=int(payload.get("ref_view", 0)),
    )
    try:
        masks = build_prompts(s.prep, spec, log=s.log)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    s.prompt_masks = masks

    R = ref()
    colors = R.object_colors_bgr(masks.shape[0])
    preview = sam_prompt.encode_mask_preview(s.prep.src.images[spec.ref_view], masks, colors)
    ok, buf = cv2.imencode(".png", preview)
    if not ok:
        raise HTTPException(500, "preview encode failed")
    rgb = R.object_colors_rgb(masks.shape[0])
    return Response(buf.tobytes(), media_type="image/png", headers={
        "Cache-Control": "no-store",
        "X-Object-Count": str(masks.shape[0]),
        "X-Object-Colors": json.dumps((rgb * 255).astype(int).tolist()),
        "X-Object-Areas": json.dumps([int(m.sum()) for m in masks]),
    })


@app.post("/api/session/{sid}/run")
def session_run(sid: str, payload: Dict = Body(...)) -> JSONResponse:
    s = store.get(sid)
    if s.prep is None:
        raise HTTPException(409, "session not ready")
    spec = PromptSpec(
        mode=payload.get("mode", "points"),
        objects=payload.get("objects", []),
        top_k=int(payload.get("top_k", 10)),
        points_per_side=int(payload.get("points_per_side", 32)),
        ref_view=int(payload.get("ref_view", 0)),
    )
    store.submit_run(
        s, spec,
        mask_threshold=float(payload.get("mask_threshold", 0.5)),
        exclusive=bool(payload.get("exclusive", True)),
        geo_cc=bool(payload.get("geo_cc", True)),
        render_mode=payload.get("render_mode", "hl"),
        do_render=bool(payload.get("render", True)),
    )
    s.log(f"[queue] segmentation queued ({store.jobs.depth()} job(s) ahead)")
    return JSONResponse(s.snapshot())


@app.get("/api/session/{sid}/file/{name}")
def session_file(sid: str, name: str) -> FileResponse:
    s = store.get(sid)
    if s.out_dir is None:
        raise HTTPException(404, "session has produced no artefacts yet")
    p = (s.out_dir / name).resolve()
    if s.out_dir.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "no such artefact")
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.post("/api/session/{sid}/viser")
def session_viser(sid: str) -> JSONResponse:
    s = store.get(sid)
    if not s.manifest or "cloud_npz" not in s.manifest or s.out_dir is None:
        raise HTTPException(409, "no point cloud yet")
    cloud = s.out_dir / str(s.manifest["cloud_npz"])
    port = viser_registry.serve(sid, cloud, label=str(s.manifest.get("name", sid[:8])))
    s.viser_port = port
    s.log(f"[viser] viewer on port {port}")
    return JSONResponse({"port": port})


@app.on_event("shutdown")
def _shutdown() -> None:
    viser_registry.stop_all()


app.mount("/static", StaticFiles(directory=str(paths.WEB_ROOT)), name="static")


def main() -> None:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--viser_base_port", type=int, default=7861)
    args = ap.parse_args()
    viser_registry.base_port = args.viser_base_port
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
