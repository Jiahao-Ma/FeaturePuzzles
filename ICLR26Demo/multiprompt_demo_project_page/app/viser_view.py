"""Interactive viser viewer: per-frame point-cloud playback with a follow camera.

The camera path is built with the same ``make_follow_path`` the offline renderer
uses, so dragging the timeline here reproduces the main view of the exported
video rather than approximating it.

Each session gets its own server on its own port; ``ViserSession`` owns that
lifecycle so the web backend can start, reuse and stop viewers per run.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import paths  # noqa: F401  (sys.path bootstrap for rgbd_render)

COLOR_MODES = ("highlight", "segmentation", "rgb")


@dataclass
class CloudData:
    xyz: np.ndarray          # (P,3)
    rgb: np.ndarray          # (P,3) float [0,1]
    seg: np.ndarray          # (P,3) float [0,1] -- object colour where masked
    obj: np.ndarray          # (P,)  int16, -1 = unmasked
    frame: np.ndarray        # (P,)  int32
    c2w: np.ndarray          # (S,4,4)
    K: np.ndarray            # (S,3,3)
    image_hw: np.ndarray     # (2,)

    @classmethod
    def load(cls, npz_path: Path) -> "CloudData":
        d = np.load(npz_path)
        return cls(xyz=d["xyz"], rgb=d["rgb"], seg=d["seg"], obj=d["obj"],
                   frame=d["frame"], c2w=d["c2w"], K=d["K"], image_hw=d["image_hw"])

    @property
    def n_frames(self) -> int:
        return int(self.c2w.shape[0])


def _follow_path(cloud: "CloudData", median_depth: float, max_fov: float = 60.0):
    """Follow camera path, using the offsets and FOV of the offline renderer."""
    from rgbd_render.camera import make_follow_path

    from .render import follow_fov, follow_offsets

    c2w = cloud.c2w
    o = follow_offsets(c2w, median_depth)
    fov = follow_fov(float(cloud.K[0][1, 1]), int(cloud.image_hw[0]),
                     median_depth, max_fov)
    return make_follow_path(
        c2w, o["scene_scale"], smooth_window=o["smooth_window"],
        back_offset=o["back_offset"], up_offset=o["up_offset"],
        look_offset=o["look_offset"],
        follow_scale=None, fov_deg=fov, keyframe_interval=1)


def _median_depth_from_cloud(cloud: CloudData) -> float:
    """Distance from each frame's camera to its own points, medianed."""
    d: List[float] = []
    for v in range(min(cloud.n_frames, 12)):
        sel = cloud.frame == v
        if not sel.any():
            continue
        c = cloud.c2w[v][:3, 3]
        d.append(float(np.median(np.linalg.norm(cloud.xyz[sel] - c, axis=1))))
    return float(np.median(d)) if d else 1.0


class ViserSession:
    """A running viser server bound to one result."""

    def __init__(self, cloud_npz: Path, port: int, host: str = "0.0.0.0",
                 label: str = "", log=print):
        import viser

        self.cloud = CloudData.load(Path(cloud_npz))
        self.port = port
        self.label = label or Path(cloud_npz).stem
        self._log = log
        self._stop = threading.Event()

        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(control_layout="collapsible",
                                        dark_mode=True)

        self.median_depth = _median_depth_from_cloud(self.cloud)
        self.path = _follow_path(self.cloud, self.median_depth)

        self._frame_nodes: Dict[int, object] = {}
        self._frustum_nodes: List[object] = []
        self._build_scene()
        self._build_gui()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        log(f"[viser] serving '{self.label}' on port {port} "
            f"({len(self.cloud.xyz):,} points, {self.cloud.n_frames} frames)")

    # -- scene -----------------------------------------------------------
    def _colors_for(self, mode: str, sel: np.ndarray) -> np.ndarray:
        if mode == "rgb":
            return self.cloud.rgb[sel]
        if mode == "segmentation":
            return self.cloud.seg[sel]
        return self.cloud.seg[sel]        # highlight: masked tinted, rest RGB

    def _build_scene(self) -> None:
        # Centre the cloud so viser's default orbit target is useful.
        self._centre = self.cloud.xyz.mean(axis=0)
        for v in range(self.cloud.n_frames):
            sel = self.cloud.frame == v
            if not sel.any():
                continue
            node = self.server.scene.add_point_cloud(
                name=f"/frames/{v:04d}",
                points=(self.cloud.xyz[sel] - self._centre).astype(np.float32),
                colors=self._colors_for("highlight", sel).astype(np.float32),
                point_size=max(self.median_depth * 0.004, 1e-4),
                point_shape="rounded",
            )
            self._frame_nodes[v] = node

    def _refresh_colors(self, mode: str) -> None:
        for v, node in self._frame_nodes.items():
            sel = self.cloud.frame == v
            if mode == "segmentation":
                # Drop unmasked points entirely rather than recolouring them.
                keep = sel & (self.cloud.obj >= 0)
                node.points = (self.cloud.xyz[keep] - self._centre).astype(np.float32)
                node.colors = self.cloud.seg[keep].astype(np.float32)
            else:
                node.points = (self.cloud.xyz[sel] - self._centre).astype(np.float32)
                node.colors = self._colors_for(mode, sel).astype(np.float32)

    # -- gui -------------------------------------------------------------
    def _build_gui(self) -> None:
        g = self.server.gui
        n = self.cloud.n_frames
        g.add_markdown(f"### {self.label}")
        with g.add_folder("Playback"):
            self.gui_frame = g.add_slider("Frame", min=0, max=max(n - 1, 1), step=1,
                                          initial_value=0)
            self.gui_playing = g.add_checkbox("Playing", True)
            self.gui_fps = g.add_slider("FPS", min=1, max=30, step=1, initial_value=8)
        with g.add_folder("View"):
            self.gui_reveal = g.add_dropdown(
                "Frames shown", ("Accumulate", "Current only", "All"),
                initial_value="Accumulate")
            self.gui_color = g.add_dropdown("Colour", COLOR_MODES,
                                            initial_value="highlight")
            self.gui_follow = g.add_checkbox("Follow camera", True)
            self.gui_point_size = g.add_slider(
                "Point size", min=0.0005, max=0.05, step=0.0005,
                initial_value=float(max(self.median_depth * 0.004, 1e-3)))
            self.gui_frustums = g.add_checkbox("Show camera frustums", False)

        self.gui_frame.on_update(lambda _: self._apply_frame())
        self.gui_reveal.on_update(lambda _: self._apply_frame())
        self.gui_follow.on_update(lambda _: self._apply_frame())
        self.gui_color.on_update(lambda _: self._refresh_colors(self.gui_color.value))
        self.gui_point_size.on_update(lambda _: self._apply_point_size())
        self.gui_frustums.on_update(lambda _: self._apply_frustums())
        self._apply_frame()

    def _apply_point_size(self) -> None:
        for node in self._frame_nodes.values():
            node.point_size = float(self.gui_point_size.value)

    def _apply_frustums(self) -> None:
        if self.gui_frustums.value and not self._frustum_nodes:
            import viser.transforms as vtf
            h, w = int(self.cloud.image_hw[0]), int(self.cloud.image_hw[1])
            for v in range(self.cloud.n_frames):
                c2w = self.cloud.c2w[v]
                fy = float(self.cloud.K[v][1, 1])
                self._frustum_nodes.append(self.server.scene.add_camera_frustum(
                    name=f"/cams/{v:04d}",
                    fov=2 * float(np.arctan2(h / 2, fy)),
                    aspect=w / h,
                    scale=self.median_depth * 0.06,
                    wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                    position=c2w[:3, 3] - self._centre,
                    color=(90, 170, 255),
                ))
        for node in self._frustum_nodes:
            node.visible = bool(self.gui_frustums.value)

    def _apply_frame(self) -> None:
        cur = int(self.gui_frame.value)
        reveal = self.gui_reveal.value
        for v, node in self._frame_nodes.items():
            if reveal == "All":
                node.visible = True
            elif reveal == "Current only":
                node.visible = (v == cur)
            else:
                node.visible = (v <= cur)
        if self.gui_follow.value:
            self._move_camera(cur)

    def _move_camera(self, frame: int) -> None:
        cam = self.path.get_camera(int(np.clip(frame, 0, self.cloud.n_frames - 1)))
        for client in self.server.get_clients().values():
            client.camera.position = (cam.eye - self._centre).astype(np.float32)
            client.camera.look_at = (cam.center - self._centre).astype(np.float32)
            client.camera.up_direction = cam.up.astype(np.float32)
            client.camera.fov = float(np.radians(cam.fov_deg))

    # -- loop ------------------------------------------------------------
    def _playback_loop(self) -> None:
        while not self._stop.is_set():
            if self.gui_playing.value:
                nxt = (int(self.gui_frame.value) + 1) % self.cloud.n_frames
                self.gui_frame.value = nxt        # triggers _apply_frame
            time.sleep(1.0 / max(float(self.gui_fps.value), 1.0))

    def stop(self) -> None:
        self._stop.set()
        try:
            self.server.stop()
        except Exception:
            pass
        self._log(f"[viser] stopped port {self.port}")


class ViserRegistry:
    """One viewer per session, on ports allocated from a fixed base."""

    def __init__(self, base_port: int = 8081, host: str = "0.0.0.0", log=print):
        self.base_port = base_port
        self.host = host
        self._log = log
        self._sessions: Dict[str, ViserSession] = {}
        self._lock = threading.Lock()

    def _free_port(self) -> int:
        used = {s.port for s in self._sessions.values()}
        p = self.base_port
        while p in used:
            p += 1
        return p

    def serve(self, key: str, cloud_npz: Path, label: str = "") -> int:
        with self._lock:
            old = self._sessions.pop(key, None)
            if old is not None:
                old.stop()
            port = self._free_port()
            self._sessions[key] = ViserSession(
                cloud_npz, port, host=self.host, label=label, log=self._log)
            return port

    def port_for(self, key: str) -> Optional[int]:
        s = self._sessions.get(key)
        return s.port if s else None

    def stop(self, key: str) -> None:
        with self._lock:
            s = self._sessions.pop(key, None)
        if s:
            s.stop()

    def stop_all(self) -> None:
        with self._lock:
            items = list(self._sessions.values())
            self._sessions.clear()
        for s in items:
            s.stop()


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True, help="*_cloud.npz from the pipeline")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    s = ViserSession(Path(args.cloud), args.port)
    print(f"viser running at http://localhost:{args.port} -- Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.stop()


if __name__ == "__main__":
    _cli()
