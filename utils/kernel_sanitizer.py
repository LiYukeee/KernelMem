# utils/kernel_sanitizer.py
"""Post-process LLM-generated kernel code to fix common hallucination patterns.

Two categories of bugs are addressed:

1. **Python-level hallucinations** – ``load_inline`` keyword arguments with
   spaces inserted into identifier names (e.g. ``cud a_source s=``) or wrong
   but similar names (e.g. ``cua_sources=``, ``extra_cua_cflags=``).

2. **CUDA/C++ identifier hallucinations** – OCR-style corruption inside string
   literals that are passed to nvcc (e.g. ``C1e_CUA_KERNEL_LAUNCH_CHECK()``,
   ``size t``, ``batch.sjze``).

3. **TORCH_LIBRARY + ``functions=`` double-registration conflict** – when
   ``cpp_sources`` contains a ``TORCH_LIBRARY`` block *and* ``functions=[...]``
   is non-empty, ``load_inline`` auto-generates a ``main.cpp`` stub that
   references the CUDA symbol from a different TU scope, causing::

       error: 'my_func' was not declared in this scope

   Fix: clear ``functions=[]`` when ``TORCH_LIBRARY`` is detected, so only the
   TORCH_LIBRARY-based registration is used.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["sanitize_kernel_code"]

# ---------------------------------------------------------------------------
# 1. load_inline Python keyword-argument typos
# ---------------------------------------------------------------------------
# Each entry is (regex_pattern, replacement).
# Patterns use \b word-boundary anchors and allow arbitrary internal whitespace
# inside the hallucinated identifier so we can catch "cud a_source s=".
_LOAD_INLINE_ARG_FIXES: list[tuple[str, str]] = [
    # ── spaced-out identifiers ─────────────────────────────────────────────
    (r'\bcud\s+a_source\s*s\s*=',          'cuda_sources='),
    (r'\bcpp_source\s+s\s*=',              'cpp_sources='),
    (r'\bfunction\s+s\s*=',               'functions='),
    (r'\bextra_cflag\s+s\s*=',            'extra_cflags='),
    (r'\bextra_cud\s+a_cflag\s+s\s*=',    'extra_cuda_cflags='),
    (r'\bextra_cud\s+a_cflags\s*=',       'extra_cuda_cflags='),
    # ── wrong but similar names (no spaces) ───────────────────────────────
    (r'\bcua_sources\s*=',                 'cuda_sources='),
    (r'\bextra_cua_cflags\s*=',            'extra_cuda_cflags='),
    # "extra_cuda_flags" (missing trailing 'c' → "cflags")
    (r'\bextra_cuda_flags\s*=',            'extra_cuda_cflags='),
    # ── Python variable names split by a space: "cud a_code" → "cuda_code" ─
    # This fixes references like `cuda_sources=[cud a_code]`
    (r'\bcud\s+a_',                        'cuda_'),
    # "__init__" argument / local variable name corruptions in Python signatures
    (r'\boutput\s+size\b',                 'output_size'),
    (r'\bs\s+elf\b',                       'self'),
    # Attribute access with space: "self.output layer" → "self.output_layer"
    (r'\bself\.output\s+layer\b',          'self.output_layer'),
    # Class name split: "class Mod elNew" → "class ModelNew"
    (r'\bMod\s+elNew\b',                   'ModelNew'),
    # "n n.ModuleList" → "nn.ModuleList"  (common nn module hallucination)
    (r'\bn\s+n\.',                          'nn.'),
    # ── gencode flag typos inside string values ───────────────────────────
    # e.g.  "compute_80,cod_e=s m _80"  →  "compute_80,code=sm_80"
    (r'(compute_\d+),cod_e\s*=\s*s\s*m\s*_(\d+)',  r'\1,code=sm_\2'),
    (r'(compute_\d+),code\s*=\s*s\s*m\s*_(\d+)',   r'\1,code=sm_\2'),
]

# ---------------------------------------------------------------------------
# 2. CUDA / C++ identifier hallucinations (apply globally to the whole file)
# ---------------------------------------------------------------------------
_CUDA_IDENT_FIXES: list[tuple[str, str]] = [
    # C10_CUDA_KERNEL_LAUNCH_CHECK variants
    (r'\bC1e_CUA_KERNEL_LAUNCH_CHECK\s*\(\)',   'C10_CUDA_KERNEL_LAUNCH_CHECK()'),
    (r'\bC10_CUA_KERNEL_LAUNCH_CHECK\s*\(\)',   'C10_CUDA_KERNEL_LAUNCH_CHECK()'),
    (r'\bC1e_CUDA_KERNEL_LAUNCH_CHECK\s*\(\)',  'C10_CUDA_KERNEL_LAUNCH_CHECK()'),
    # C++ namespace with spaces: "at :: cud a ::" → "at::cuda::"
    (r'\bat\s*::\s*cud\s+a\s*::\s*',              'at::cuda::'),
    (r'\bat\s*::\s*cuda\s*::\s*',                 'at::cuda::'),
    # "size t" → "size_t"
    (r'\bsize\s+t\b',                           'size_t'),
    # "return out put ;" → "return output;"
    (r'\breturn\s+out\s+put\s*;',               'return output;'),
    # kernel launch with space: "foo_bar kernel<<<" → "foo_bar_kernel<<<"
    (r'(\w+)\s+kernel\s*<<<',                   r'\1_kernel<<<'),
    # OCR-style variable name corruptions (seen in real V3 outputs)
    (r'\bbatch\.sjze\b',                        'batch_size'),
    (r'\boutput_se\b',                          'output_size'),
    (r'\bbatch_jze\b',                          'batch_size'),
    (r'\byput\s+size\b',                        'output_size'),
    (r'\bshmen\s*_\s*siie\b',                   'shmem_size'),
    (r'\bshmem\._aze\b',                        'shmem_size'),
    (r'\boutpu3\b',                             'output'),
]

# ---------------------------------------------------------------------------
# 3. TORCH_LIBRARY + functions=[] double-registration
# ---------------------------------------------------------------------------
_TORCH_LIBRARY_RE = re.compile(r'TORCH_LIBRARY\s*\(')

# Matches the `functions=` kwarg in a load_inline() call, capturing the list.
# Handles multi-line lists by matching balanced brackets conservatively.
_FUNCTIONS_ARG_RE = re.compile(
    r'(functions\s*=\s*)(\[[^\]]*\])',
    re.DOTALL,
)

# Matches the string content of cpp_sources=[ "..." ] (single or multi-line).
_CPP_SOURCES_CONTENT_RE = re.compile(
    r'cpp_sources?\s*=\s*\[([^\]]*)\]',
    re.DOTALL,
)


def _cpp_sources_contain_torch_library(code: str) -> bool:
    """Return True if any cpp_sources string contains TORCH_LIBRARY."""
    for m in _CPP_SOURCES_CONTENT_RE.finditer(code):
        if _TORCH_LIBRARY_RE.search(m.group(1)):
            return True
    # Fallback: the whole code has TORCH_LIBRARY anywhere
    return bool(_TORCH_LIBRARY_RE.search(code))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fix_load_inline_args(code: str) -> str:
    """Fix hallucinated / misspelled ``load_inline`` keyword argument names."""
    for pattern, replacement in _LOAD_INLINE_ARG_FIXES:
        before = code
        code = re.sub(pattern, replacement, code)
        if code != before:
            logger.debug("[sanitizer] load_inline fix applied: %r", pattern)
    return code


def fix_cuda_identifiers(code: str) -> str:
    """Fix hallucinated CUDA/C++ identifiers (spaces inside names, OCR errors)."""
    for pattern, replacement in _CUDA_IDENT_FIXES:
        before = code
        code = re.sub(pattern, replacement, code)
        if code != before:
            logger.debug("[sanitizer] cuda ident fix applied: %r", pattern)
    return code


def fix_double_registration(code: str) -> str:
    """Clear ``functions=[]`` when ``TORCH_LIBRARY`` is present in cpp_sources.

    This prevents the auto-generated ``main.cpp`` stub from referencing a CUDA
    symbol that is only visible inside the ``.cu`` translation unit, which
    causes ``'func' was not declared in this scope`` compile errors.
    """
    if _cpp_sources_contain_torch_library(code):
        before = code
        code = _FUNCTIONS_ARG_RE.sub(r'\1[]', code)
        if code != before:
            logger.debug("[sanitizer] Cleared functions=[] due to TORCH_LIBRARY conflict")
    return code


def sanitize_kernel_code(code: str) -> str:
    """Apply all sanitization passes and return cleaned kernel code.

    Pass order:
      0. Normalize non-standard whitespace (U+00A0 etc.) to ASCII space
      1. Fix ``load_inline`` keyword-argument name hallucinations (Python level)
      2. Fix CUDA/C++ identifier hallucinations (applies to whole file text)
      3. Clear ``functions=[]`` when ``TORCH_LIBRARY`` double-registration is
         detected (avoids the most common CompilationError pattern)

    The function is intentionally conservative: it only rewrites text that
    matches highly specific patterns from observed V3 outputs, never touching
    semantically correct code.
    """
    # Pass 0: normalize non-ASCII whitespace that causes tokenization errors
    code = code.replace('\u00a0', ' ')   # non-breaking space → regular space
    code = code.replace('\u200b', '')    # zero-width space → remove
    code = code.replace('\u2019', "'")   # right single quotation → apostrophe

    code = fix_load_inline_args(code)
    code = fix_cuda_identifiers(code)
    code = fix_double_registration(code)
    return code
