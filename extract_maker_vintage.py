# wine_fields.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

ROOT = Path(__file__).resolve().parent

# --- robust import of run_once from ocr_scripts/run_ocr.py ---
try:
    # Preferred: treat ocr_scripts as a package
    from ocr_scripts.run_ocr import run_once
except Exception:
    # Fallback: add the folder to sys.path if not a package
    OCR_DIR = ROOT / "ocr_scripts"
    if str(OCR_DIR) not in sys.path:
        sys.path.insert(0, str(OCR_DIR))
    from run_ocr import run_once  # type: ignore

# Default weights next to repo root (adjust if yours lives elsewhere)
DEFAULT_WEIGHTS = ROOT / "weights.pt"

# ---- helpers (copied from scan_and_store.py) ----
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 &'-")

def normalize_maker(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.upper()
    s = "".join(ch for ch in s if ch in ALLOWED)
    s = " ".join(s.split())
    return s

def extract_best_maker_from_raw(raw: Dict[str, Any]) -> str:
    candidates = []
    for txt, _score in raw.get("maker_name_candidates", []):
        if txt:
            candidates.append(txt)
    for k, arr in raw.items():
        if k == "maker_name_candidates":
            continue
        for txt, _score in arr:
            if isinstance(txt, str) and len(txt.strip()) >= 3 and txt.strip() != ".":
                candidates.append(txt)
    if not candidates:
        return ""
    candidates = [normalize_maker(t) for t in candidates if t]
    candidates = [t for t in candidates if t]
    return max(candidates, key=len) if candidates else ""

def _to_int_year(v: Optional[Union[str, int]]) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    return int(s) if (len(s) == 4 and s.isdigit()) else None

# ---- public API ----
def wine_fields(
    image_path: Union[str, Path],
    weights_path: Optional[Union[str, Path]] = None
) -> List[Optional[Union[str, int]]]:
    """
    Returns [CustomID, MakerName, Vintage]
    - CustomID: "MakerName|Vintage" when both exist, else None
    - MakerName: normalized uppercase string (or None)
    - Vintage: 4-digit int (or None)
    """
    w = str(weights_path or DEFAULT_WEIGHTS)
    res = run_once(str(image_path), w) or {}

    maker_raw = res.get("maker_name")
    raw = res.get("raw") or {}

    maker_norm = normalize_maker(maker_raw) if maker_raw else ""
    if not maker_norm:
        maker_norm = extract_best_maker_from_raw(raw)
    maker_out = maker_norm or None

    vintage_int = _to_int_year(res.get("vintage"))

    custom_id = f"{maker_out}|{vintage_int}" if (maker_out and vintage_int) else None
    return [custom_id, maker_out, vintage_int]

# Optional CLI
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python wine_fields.py <image_path> [weights_path]")
        sys.exit(1)
    img = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) >= 3 else None
    print(json.dumps(wine_fields(img, weights), indent=2))
