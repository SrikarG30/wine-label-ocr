import os, sys, json, hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from rapidfuzz import fuzz
from run_ocr import run_once

DB_PATH_DEFAULT = "current_cellar.jsonl"

ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 &'-")

def normalize_maker(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.upper()
    s = "".join(ch for ch in s if ch in ALLOWED)
    s = " ".join(s.split())
    return s

def extract_best_maker_from_raw(raw: Dict[str, Any]) -> str:
    # first try maker name candidates
    candidates = []
    for txt, _score in raw.get("maker_name_candidates", []):
        if txt:
            candidates.append(txt)

    # fallback: scan other raw buckets
    for k, arr in raw.items():
        if k == "maker_name_candidates":
            continue
        for txt, _score in arr:
            if isinstance(txt, str) and len(txt.strip()) >= 3 and txt.strip() != ".":
                candidates.append(txt)

    if not candidates:
        return ""

    candidates = [normalize_maker(t) for t in candidates if t]
    candidates = [t for t in candidates if t]  # remove empty entries
    if not candidates:
        return ""

    # pick the longest normalized candidate as a simple heuristic
    return max(candidates, key=len)

def canonical_key(maker_norm: str, vintage: Optional[str]) -> str:
    return f"{maker_norm}|{vintage or ''}"

def hash_id(maker_norm: str, vintage: Optional[str]) -> str:
    m = hashlib.sha1()
    m.update(canonical_key(maker_norm, vintage).encode("utf-8"))
    return m.hexdigest()[:12]

# ---------- storage backends ----------
def _is_txt(path: str) -> bool:
    return path.lower().endswith(".txt")

def load_db(path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of items with fields at least:
      - maker_norm (str)
      - vintage (str or None)
      - key (str)
      - id (str)
    For .jsonl: reads your full records.
    For .txt: each non-empty line is a key "MAKER|VINTAGE".
    """
    if not os.path.exists(path):
        return []

    items: List[Dict[str, Any]] = []
    if _is_txt(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if not key or key.startswith("#"):
                    continue
                if "|" in key:
                    maker_norm, vintage = key.split("|", 1)
                else:
                    maker_norm, vintage = key, ""
                items.append({
                    "maker_norm": maker_norm,
                    "vintage": vintage or None,
                    "key": key,
                    "id": hash_id(maker_norm, vintage or None)
                })
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    it = json.loads(line)
                except Exception:
                    continue
                # ensure required fields exist
                maker_norm = it.get("maker_norm", "") or ""
                vintage = it.get("vintage")
                it["maker_norm"] = maker_norm
                it["vintage"] = vintage
                it["key"] = it.get("key") or canonical_key(maker_norm, vintage)
                it["id"] = it.get("id") or hash_id(maker_norm, vintage)
                items.append(it)
    return items

def append_db(path: str, record: Dict[str, Any]) -> None:
    """
    For .txt: append the canonical key only.
    For .jsonl: append the full record as JSON.
    """
    if _is_txt(path):
        key = record["key"]
        with open(path, "a", encoding="utf-8") as f:
            f.write(key + "\n")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------- Matching ----------
def best_match(
    db: List[Dict[str, Any]],
    maker_norm: str,
    vintage: Optional[str],
    maker_threshold: int = 85
):
    """
    Returns (decision, best_item, best_score, reason)
    decision: 'seen' or 'not seen'
    """
    if not db:
        return "not seen", None, 0, "db empty"

    key = canonical_key(maker_norm, vintage)

    # exact canonical key match (strongest)
    for it in db:
        if it.get("key") == key and key != "|":
            return "seen", it, 100, "exact canonical key match"

    # fuzzy maker match (prefer same vintage if present)
    cands = db
    if vintage:
        same_vintage = [it for it in db if it.get("vintage") == vintage]
        if same_vintage:
            cands = same_vintage

    best_item = None
    best_score = -1

    for it in cands:
        other_maker = it.get("maker_norm", "")
        score = fuzz.token_set_ratio(maker_norm, other_maker)
        if vintage and it.get("vintage") == vintage:
            score = min(100, score + 5) 
        if score > best_score:
            best_score = score
            best_item = it

    if best_score >= maker_threshold and best_item:
        return "seen", best_item, best_score, f"fuzzy maker match >= {maker_threshold}"

    return "not seen", best_item, best_score, "no sufficient match"

def build_record(image_path: str, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    maker = ocr_result.get("maker_name")
    vintage = ocr_result.get("vintage")
    raw = ocr_result.get("raw", {}) or {}

    maker_norm = normalize_maker(maker) if maker else ""
    if not maker_norm:
        maker_norm = extract_best_maker_from_raw(raw)

    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "maker_name": maker or None,
        "maker_norm": maker_norm or "",
        "vintage": vintage or None,
        "key": canonical_key(maker_norm, vintage),
        "id": hash_id(maker_norm, vintage),
        "ocr_raw": raw
    }
    return rec

def main():
    if len(sys.argv) < 4:
        print("Usage: python scan_and_store.py <image_path> <weights_path> <store_path(.jsonl or .txt)>")
        sys.exit(1)

    image_path = sys.argv[1]
    weights_path = sys.argv[2]
    store_path = sys.argv[3]

    ocr_result = run_once(image_path, weights_path)
    record = build_record(image_path, ocr_result)

    # basic guard to avoid storing empty keys like "|"
    if record["key"] == "|" or (not record["maker_norm"] and not record["vintage"]):
        print(json.dumps({
            "decision": "not stored",
            "reason": "empty/invalid maker & vintage",
            "current": {
                "maker_norm": record["maker_norm"],
                "vintage": record["vintage"],
                "key": record["key"]
            }
        }, indent=2))
        sys.exit(0)

    db = load_db(store_path)
    decision, best_item, score, reason = best_match(
        db, record["maker_norm"], record["vintage"], maker_threshold=85
    )

    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "score": score,
        "current": {
            "maker_norm": record["maker_norm"],
            "vintage": record["vintage"],
            "key": record["key"],
            "id": record["id"]
        },
        "match": {
            "id": best_item.get("id") if best_item else None,
            "maker_norm": best_item.get("maker_norm") if best_item else None,
            "vintage": best_item.get("vintage") if best_item else None,
            "key": best_item.get("key") if best_item else None
        } if best_item else None
    }, indent=2))

    if decision == "not seen":
        append_db(store_path, record)

if __name__ == "__main__":
    main()
