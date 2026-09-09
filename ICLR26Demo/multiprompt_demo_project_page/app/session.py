"""Per-session state and the single-slot GPU job queue.

Only one segmentation job can hold the GPU at a time, so jobs are serialised
through one worker thread. Sessions keep a rolling log that the browser polls
for progress.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import paths
from .pipeline import PreparedInput, PromptSpec, prepare, run, unique_run_dir

MAX_LOG_LINES = 400


@dataclass
class Session:
    sid: str
    source: Path
    scene_mode: str
    n_frames: int = 24
    # Assigned when a segmentation run starts; browsing a clip writes nothing.
    out_dir: Optional[Path] = None
    created: float = field(default_factory=time.time)

    prep: Optional[PreparedInput] = None
    prompt_masks: Optional[np.ndarray] = None
    manifest: Optional[Dict] = None
    viser_port: Optional[int] = None

    state: str = "new"          # new | preparing | ready | running | done | error
    error: Optional[str] = None
    progress: str = ""
    log_lines: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, msg: str) -> None:
        line = str(msg).rstrip()
        if not line:
            return
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > MAX_LOG_LINES:
                del self.log_lines[:-MAX_LOG_LINES]
            self.progress = line
        print(f"[{self.sid[:8]}] {line}", flush=True)

    def dump_log(self) -> None:
        """Persist the run log next to the artefacts it describes.

        The in-memory log is capped and the browser only ever sees its tail, so
        without this the reasoning behind a saved result is lost on restart.
        """
        if self.out_dir is None:
            return
        with self._lock:
            lines = list(self.log_lines)
        try:
            (self.out_dir / "session_log.txt").write_text("\n".join(lines) + "\n")
        except OSError as exc:
            print(f"[{self.sid[:8]}] could not write session log: {exc}", flush=True)

    def snapshot(self) -> Dict:
        with self._lock:
            lines = list(self.log_lines)
        return {
            "sid": self.sid,
            "state": self.state,
            "error": self.error,
            "progress": self.progress,
            "log": lines[-60:],
            "scene_mode": self.scene_mode,
            "source": self.source.name,
            "n_frames": int(self.prep.src.size) if self.prep else self.n_frames,
            "frame_hw": list(self.prep.src.hw) if self.prep else None,
            "geometry": (("ground-truth" if self.prep.src.has_gt_geometry else "predicted")
                         if self.prep else None),
            "n_objects": int(self.prompt_masks.shape[0]) if self.prompt_masks is not None else 0,
            "manifest": self.manifest,
            "viser_port": self.viser_port,
            "out_dir": self.out_dir.name if self.out_dir else None,
        }


class JobQueue:
    """Serialises GPU work onto one worker thread."""

    def __init__(self, log=print):
        self._q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._log = log
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], None]) -> None:
        self._q.put(fn)

    def depth(self) -> int:
        return self._q.qsize()

    def _loop(self) -> None:
        while True:
            fn = self._q.get()
            try:
                fn()
            except Exception:
                self._log("[job] unhandled error:\n" + traceback.format_exc())
            finally:
                self._q.task_done()


class SessionStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or paths.ensure_runs()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.jobs = JobQueue()

    def create(self, source: Path, scene_mode: str, n_frames: int) -> Session:
        sid = uuid.uuid4().hex
        s = Session(sid=sid, source=Path(source), scene_mode=scene_mode,
                    n_frames=n_frames)
        with self._lock:
            self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Session:
        with self._lock:
            s = self._sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        return s

    def all(self) -> List[Session]:
        with self._lock:
            return list(self._sessions.values())

    # -- queued work -----------------------------------------------------
    def submit_prepare(self, s: Session, long_side: int = 640) -> None:
        def job() -> None:
            s.state = "preparing"
            try:
                s.prep = prepare(s.source, s.scene_mode,
                                 n_frames=s.n_frames, long_side=long_side, log=s.log)
                s.state = "ready"
                s.log(f"[ready] {s.prep.src.size} frames, "
                      f"{'ground-truth' if s.prep.src.has_gt_geometry else 'predicted'} geometry")
            except Exception as exc:
                s.state = "error"
                s.error = f"{type(exc).__name__}: {exc}"
                s.log(f"[error] prepare failed: {s.error}")
                s.log(traceback.format_exc())
        self.jobs.submit(job)

    def submit_run(self, s: Session, spec: PromptSpec, **kw) -> None:
        def job() -> None:
            if s.prep is None:
                s.state = "error"
                s.error = "session not prepared"
                return
            s.state = "running"
            s.manifest = None
            try:
                # A fresh directory per run, named after the clip: prompting the
                # same clip twice keeps both results instead of overwriting.
                s.out_dir = unique_run_dir(self.root, s.prep.run_name)
                s.log(f"[out] {s.out_dir.name}/")
                s.manifest = run(s.prep, spec, s.out_dir,
                                 name=s.prep.run_name, log=s.log, **kw)
                s.state = "done"
            except Exception as exc:
                s.state = "error"
                s.error = f"{type(exc).__name__}: {exc}"
                s.log(f"[error] run failed: {s.error}")
                s.log(traceback.format_exc())
            finally:
                s.dump_log()
        self.jobs.submit(job)
