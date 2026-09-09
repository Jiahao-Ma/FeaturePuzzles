"""Access to the validated static-scene pipeline.

``render_multiprompt_demo.py`` in ``vggt_omega_multiprompt_static_scene`` is the
reference implementation that produced the ICLR static-scene results. Rather
than fork it, the demo loads it as a module and reuses its segmentation,
geometric post-filter, point-collection and rendering helpers verbatim, so the
web demo and the paper numbers come from exactly the same code.
"""
from __future__ import annotations

import importlib.util
from types import ModuleType

from . import paths  # noqa: F401  (bootstraps sys.path)

_STATIC_DEMO = paths.DEMO_ROOT.parent / "vggt_omega_multiprompt_static_scene" / "render_multiprompt_demo.py"

_module: ModuleType | None = None


def ref() -> ModuleType:
    """Import (once) and return the static-scene reference module."""
    global _module
    if _module is None:
        if not _STATIC_DEMO.exists():
            raise FileNotFoundError(f"static reference pipeline not found: {_STATIC_DEMO}")
        spec = importlib.util.spec_from_file_location("_static_multiprompt_ref", _STATIC_DEMO)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _module = mod
    return _module
