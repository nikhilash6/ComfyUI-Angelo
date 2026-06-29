"""Angelo — SAM 3 post-install fixups.

Run by install_sam3_support.{bat,sh} after SAM 3 is installed/cloned, and
also on the "already installed" path so existing installs get fixed too.
Idempotent and safe to run repeatedly; always exits 0 so it can never block
the installer.

Currently patches one upstream SAM 3.1 bug
-------------------------------------------
`sam3/perflib/fused.py::addmm_act` casts its inputs to bfloat16 and returns
a bfloat16 tensor *unconditionally*. Angelo builds SAM 3 in float32 (its
known-good precision — see angelo_segment.py::_ensure_model), so that bf16
output then collides with the fp32 weights one layer downstream:

    "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"

The fix records the caller's input dtype and casts addmm_act's result back
to it — a no-op when SAM 3 genuinely runs in bf16. `vitdet.py` binds the
symbol at import time (`from sam3.perflib.fused import addmm_act`), so a
runtime monkey-patch wouldn't reach that reference; the file on disk has to
be edited. Reported in Angelo issue #31.
"""

from __future__ import annotations

import os
import sys

_MARKER = "angelo-dtype-fix"

# The exact upstream (SAM 3.1) function we expect to find. If upstream
# changes this, the patch no-ops loudly rather than corrupting the file.
_BROKEN = (
    "def addmm_act(activation, linear, mat1):\n"
    "    if torch.is_grad_enabled():\n"
    '        raise ValueError("Expected grad to be disabled.")\n'
    "    self = linear.bias.detach()\n"
    "    mat2 = linear.weight.detach()\n"
    "    self = self.to(torch.bfloat16)\n"
    "    mat1 = mat1.to(torch.bfloat16)\n"
    "    mat2 = mat2.to(torch.bfloat16)\n"
    "    mat1_flat = mat1.view(-1, mat1.shape[-1])\n"
    "    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],))\n"
    "    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],))\n"
    "    raise ValueError(f\"Unexpected activation {activation}\")"
)

_FIXED = (
    "def addmm_act(activation, linear, mat1):  # " + _MARKER + "\n"
    "    if torch.is_grad_enabled():\n"
    '        raise ValueError("Expected grad to be disabled.")\n'
    "    out_dtype = mat1.dtype  # " + _MARKER + ": preserve caller dtype (SAM 3 runs fp32)\n"
    "    self = linear.bias.detach()\n"
    "    mat2 = linear.weight.detach()\n"
    "    self = self.to(torch.bfloat16)\n"
    "    mat1 = mat1.to(torch.bfloat16)\n"
    "    mat2 = mat2.to(torch.bfloat16)\n"
    "    mat1_flat = mat1.view(-1, mat1.shape[-1])\n"
    "    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)\n"
    "    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)\n"
    "    raise ValueError(f\"Unexpected activation {activation}\")"
)


def _candidate_fused_paths():
    """Yield possible locations of sam3/perflib/fused.py, best-guess first.

    The installer clones into <this dir>/sam3, so that path covers the
    normal flow exactly. A light importlib fallback catches the case where
    sam3 was pip-installed elsewhere (vendored copy, site-packages)."""
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "sam3", "sam3", "perflib", "fused.py")

    # Fallback: locate the installed package without importing it (importing
    # sam3 pulls in torch + the whole model stack, which we don't need here).
    try:
        import importlib.util
        spec = importlib.util.find_spec("sam3")
        if spec is not None:
            for loc in (spec.submodule_search_locations or []):
                yield os.path.join(loc, "perflib", "fused.py")
    except Exception:
        pass


def _patch_fused(path):
    """Apply the dtype fix to one fused.py. Returns a short status string."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        return f"unreadable ({e})"

    if _MARKER in src:
        return "already patched"
    if _BROKEN not in src:
        return ("skipped — addmm_act doesn't match the known-broken SAM 3.1 "
                "version (upstream may have fixed it; nothing to do)")

    patched = src.replace(_BROKEN, _FIXED, 1)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patched)
    except OSError as e:
        return f"FAILED to write ({e})"
    return "patched"


def main():
    seen = set()
    patched_any = False
    found_any = False
    for path in _candidate_fused_paths():
        path = os.path.normpath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        found_any = True
        status = _patch_fused(path)
        print(f"[Angelo/SAM3 post-install] {path}: {status}")
        if status in ("patched", "already patched"):
            patched_any = True

    if not found_any:
        print("[Angelo/SAM3 post-install] no sam3/perflib/fused.py found "
              "(nothing to patch — fine if SAM 3 isn't installed yet).")
    elif not patched_any:
        print("[Angelo/SAM3 post-install] note: SAM 3's addmm_act looked "
              "different from the version we patch. If Detect later crashes "
              "with a 'BFloat16 and Float' dtype error, see issue #31.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
