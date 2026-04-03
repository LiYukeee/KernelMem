# utils/kernel_io.py
"""Utility helpers for Mind‑Evolution CUDA‑kernel workflow.

This tiny module centralizes two common I/O helpers that were previously
inlined in the end‑to‑end test script:

1. ``extract_code_block`` – extract the first ```python ... ``` (or generic) code
   block from LLM output. Raises if none found.
2. ``save_kernel_code`` – writes extracted code to *kernels/* with a unique
   timestamped filename and returns the *Path* object.

Keeping them here avoids duplication across evolution loops / diagnostics.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Final
import json
from typing import Any, Dict, List
__all__: Final = [
    "extract_code_block",
    "save_kernel_code",
]

# ---------------------------------------------------------------------------
# 1. Code‑block extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


# Match code‑fence opening; language tag is optional
_CODE_FENCE_OPEN_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n?")

def extract_code_block(text: str) -> str:
    """Return the **first** triple‑back‑ticked block in *text*.

    - After finding an opening fence, search for the closing fence. If none is
      found, consume until end of string.
    - If the text contains no ``` fences at all, raise and dump the raw output
      to a timestamped file for debugging.
    """
    if text is None:
        text = ""

    m_open = _CODE_FENCE_OPEN_RE.search(text)
    if not m_open:
        # No ``` found → raise and persist raw output to disk
        dump_path = f"llm_output_error_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(dump_path, "w") as f:
            f.write(text)
        raise RuntimeError(f"No ``` code block found in LLM output – raw output saved to {dump_path}")

    start = m_open.end()
    m_close = re.search(r"```", text[start:])
    if m_close:
        end = start + m_close.start()
        block = text[start:end]
    else:
        # No closing fence: take everything to the end
        block = text[start:]

    return block.strip() + "\n"



# ---------------------------------------------------------------------------
# 2. Persist kernel to file
# ---------------------------------------------------------------------------

def save_kernel_code(code: str, out_dir: Path | str = "kernels") -> Path:
    """Save *code* to *out_dir/kernel_YYYYmmdd_HHMMSS.py* and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")

    return path


# utils/kernel_io.py



def _repair_json_str(s: str) -> str:
    """Apply heuristic repairs to LLM-generated JSON strings.

    Handles:
    - Backslash-escaped outer quotes: ``"key": \\"value\\"``
    - Trailing commas before ``}`` or ``]``
    """
    # 1. Unescape backslash-quoted values: "key": \"value\" → "key": "value"
    s = re.sub(r'(?<=[:{,\[\s])\\"((?:[^"\\]|\\.)*)\\"\s*(?=[,}\]])',
               lambda m: '"' + m.group(1) + '"', s)
    # 2. Remove trailing commas inside objects/arrays.
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def _close_json(s: str) -> str:
    """Add missing closing brackets/braces to a truncated JSON string.

    Walks the string character-by-character (respecting string literals and
    escape sequences) and appends whatever closing chars are needed.
    """
    stack: list = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in '{[':
                stack.append('}' if ch == '{' else ']')
            elif ch in '}]' and stack and stack[-1] == ch:
                stack.pop()
    return s + ''.join(reversed(stack))


def _try_parse(candidate: str):
    """Try json.loads with three escalating repair strategies.

    1. Direct parse
    2. After _repair_json_str (unescape + trailing-comma removal)
    3. After repair + truncation at the first error position + bracket closing

    Returns the parsed Python object, or None if all strategies fail.
    """
    # Pass 1: direct
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Pass 2: after basic repair
    repaired = _repair_json_str(candidate)
    err_pos = len(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        err_pos = e.pos

    # Pass 3: truncate at the error position, close any open brackets.
    # This handles cases like arrays that contain bare "key": value pairs:
    #   "__FIELD__": [{...}, "key": value]
    # After truncation (dropping "key": value) the array + outer object close cleanly.
    trunc = repaired[:err_pos].rstrip()
    if trunc.endswith(','):
        trunc = trunc[:-1].rstrip()
    try:
        return json.loads(_close_json(trunc))
    except json.JSONDecodeError:
        pass

    return None


def _normalize_strategy(obj: Any) -> Any:
    """Normalise a parsed strategy dict.

    - Renames non-standard LLM keys (e.g. ``__EXPECTED_METRIC_CHANGE__``) to
      the canonical names expected by the rest of the code.
    - Converts list/dict field values to plain strings for fields where
      the schema expects a string.
    """
    if not isinstance(obj, dict):
        return obj

    # Rename non-standard keys produced by some LLM responses
    _aliases = {
        "__EXPECTED_METRIC_CHANGE__": "expected_metric_change",
        "modification_plan": "modification plan",
    }
    for old_key, new_key in _aliases.items():
        if old_key in obj and new_key not in obj:
            obj[new_key] = obj.pop(old_key)

    # Stringify list/dict values for fields that expect a plain string
    _flatten = {
        "modification plan", "modification_plan",
        "evidence", "expected_metric_change",
        "primary_optimisation_method",
    }
    for key in list(obj.keys()):
        if key in _flatten:
            val = obj[key]
            if isinstance(val, list):
                obj[key] = "\n".join(str(v) for v in val)
            elif isinstance(val, dict):
                obj[key] = json.dumps(val, ensure_ascii=False)
    return obj


def extract_json(raw: str) -> Any:
    """
    Extract the first JSON object/array from a string and parse it into a Python object.
    Supports fenced code blocks like ```json ...``` or raw JSON embedded in text.
    Applies heuristic repairs for common LLM JSON formatting errors.

    Args:
        raw: Raw LLM output text.
    Returns:
        A Python object (``dict`` or ``list``).
    Raises:
        ValueError: If no valid JSON can be found/parsed.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # Try the ```json ...``` fenced format first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            return _normalize_strategy(result)

    # Try matching the first { ... } or [ ... ] (greedy — captures outermost)
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        candidate = match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            return _normalize_strategy(result)

    # Fallback: attempt to parse the whole string
    result = _try_parse(raw.strip())
    if result is not None:
        return _normalize_strategy(result)

    raise ValueError(f"Failed to extract valid JSON from reply:\n{raw}")

def save_prompt_text(text: str, out_dir: Path, *, tag: str = "repair") -> Path:
    """
    Save *text* to ``out_dir/{tag}_YYYYMMDD-HHMMSS.txt`` and return the Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{tag}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path

def extract_cuda_kernel_names(py_path: Path) -> List[str]:
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    p1 = re.compile(r"""__global__\s+void\s+([A-Za-z_]\w*)\s*\(""", re.MULTILINE)
    p2 = re.compile(
        r"""__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(""",
        re.MULTILINE,
    )

    names = p1.findall(src) + p2.findall(src)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered
