"""Training-free prompt-driven patch matching on top of VGGT / FastVGGT.

Public entry points are exposed here so callers can do:

    from prompt_matching import (
        extract_patch_tokens,
        encode_mask_prompt,
        encode_bbox_prompt,
        compute_similarity,
        binarize,
        upsample_patch_map,
    )

The heavy lifting lives in the individual modules. See ``run.py`` for the
end-to-end CLI.
"""

from .extractor import extract_patch_tokens  # noqa: F401
from .prompt import (  # noqa: F401
    encode_bbox_prompt,
    encode_bbox_prompt_soft,
    encode_bbox_prompt_subpatch,
    encode_mask_prompt,
    encode_mask_prompt_soft,
    encode_mask_prompt_subpatch,
)
from .matcher import (  # noqa: F401
    binarize,
    compute_similarity,
    compute_similarity_subpatch,
    upsample_patch_map,
)

__all__ = [
    "extract_patch_tokens",
    "encode_mask_prompt",
    "encode_mask_prompt_soft",
    "encode_mask_prompt_subpatch",
    "encode_bbox_prompt",
    "encode_bbox_prompt_soft",
    "encode_bbox_prompt_subpatch",
    "compute_similarity",
    "compute_similarity_subpatch",
    "binarize",
    "upsample_patch_map",
]
