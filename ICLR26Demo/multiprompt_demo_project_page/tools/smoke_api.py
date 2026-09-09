#!/usr/bin/env python3
"""Drive the demo's HTTP API end to end, the way the browser does.

    python tools/smoke_api.py --source scannet/scene0046_00 --scene_mode static \
        --prompt dense --top_k 6
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7860"


def post(path: str, body: dict) -> tuple[dict, dict]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        headers = dict(r.headers)
        raw = r.read()
    try:
        return json.loads(raw), headers
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"bytes": len(raw)}, headers      # image responses (mask preview)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def wait_for(sid: str, target: set[str], timeout: int = 1800) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        s = get(f"/api/session/{sid}")
        if s["progress"] != last:
            last = s["progress"]
            print(f"  · {last}")
        if s["state"] in target:
            return s
        time.sleep(1.5)
    raise TimeoutError(f"stuck in {s['state']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--scene_mode", default="static")
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--prompt", choices=["dense", "points"], default="dense")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--point", nargs=2, type=float, default=None,
                    metavar=("X", "Y"), help="normalised click, e.g. 0.47 0.50")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    print(f"[1] create session: {args.source} ({args.scene_mode})")
    s, _ = post("/api/session", {"source": args.source, "scene_mode": args.scene_mode,
                                 "n_frames": args.n_frames})
    sid = s["sid"]
    s = wait_for(sid, {"ready", "error"})
    if s["state"] == "error":
        raise SystemExit(f"prepare failed: {s['error']}")
    print(f"    ready: {s['n_frames']} frames, {s['geometry']} geometry")

    prompt = {"mode": args.prompt, "ref_view": 0, "top_k": args.top_k,
              "objects": ([{"points": [{"x": args.point[0], "y": args.point[1], "label": 1}]}]
                          if args.point else [])}

    print(f"[2] prompt ({args.prompt})")
    _, h = post(f"/api/session/{sid}/prompt", prompt)
    print(f"    {h.get('X-Object-Count')} object(s), areas={h.get('X-Object-Areas')}")

    print("[3] segment")
    post(f"/api/session/{sid}/run", {**prompt, "render": not args.no_render})
    s = wait_for(sid, {"done", "error"})
    if s["state"] == "error":
        raise SystemExit(f"run failed: {s['error']}")
    print(json.dumps(s["manifest"], indent=2))

    print("[4] viser")
    try:
        print(f"    port {post(f'/api/session/{sid}/viser', {})[0]['port']}")
    except urllib.error.HTTPError as e:
        print(f"    unavailable: {e}")
    print(f"\nartefacts: runs/{s['out_dir']}")


if __name__ == "__main__":
    main()
