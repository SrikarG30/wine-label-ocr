import re
import unicodedata
from typing import Optional

# record = {
#     'CustomID': None,       # string ('MakerName|Vintage')
#     'MakerName': None,      # string
#     'Vintage': None,        # integer (4 digits) (can also be string)
#     'Barcode': None,        # string
#     'BlobData': {}          # json (coordinates)
# }


# --- scorer (rapidfuzz) ---
try:
    from rapidfuzz import fuzz
    _HAS_RF = True
except Exception:
    _HAS_RF = False
    import difflib


def _score(a: str, b: str, method: str = "token_set") -> int:
    if _HAS_RF:
        if method == "token_set":
            return int(fuzz.token_set_ratio(a, b))
        if method == "token_sort":
            return int(fuzz.token_sort_ratio(a, b))
        if method == "partial":
            return int(fuzz.partial_ratio(a, b))
        return int(fuzz.ratio(a, b))
    return int(round(100 * difflib.SequenceMatcher(None, a, b).ratio()))


# --- normalization ---
_WINERY_WORDS = {
    "winery", "vineyard", "vineyards", "cellar", "cellars", "estate", "the", "co", "inc", "llc", "ltd",
    "domaine", "domaines", "chateau", "bodega", "bodegas", "weingut", "azienda", "tenuta", "cantina"
}


def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _normalize(s: Optional[str], strip_words: Optional[set[str]] = None) -> str:
    if not s:
        return ""
    s = _ascii_fold(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)      # remove punctuation
    toks = [t for t in s.split() if t]
    if strip_words:
        toks = [t for t in toks if t not in strip_words]
    return " ".join(toks)

# --- generic similarity ---


def strings_similar(a: Optional[str], b: Optional[str], *,
                    threshold: int = 90,
                    method: str = "token_set",
                    strip_common_winery_words: bool = False) -> bool:
    if not a or not b:
        return False
    sw = _WINERY_WORDS if strip_common_winery_words else None
    na = _normalize(a, sw)
    nb = _normalize(b, sw)
    if not na or not nb:
        return False
    return _score(na, nb, method) >= threshold


def _extract_year(s: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else None

# --- field wrappers ---


def isMakerNameSimilar(r1: dict, r2: dict, *, threshold: int = 85) -> bool:
    m1 = r1['MakerName']
    m2 = r2['MakerName']
    return strings_similar(m1, m2, threshold=threshold, strip_common_winery_words=False)


def isCustomIDSimilar(r1: dict, r2: dict, *, threshold: int = 85) -> bool:
    c1 = r1['CustomID']
    c2 = r2['CustomID']

    # Hard rule: if both have a 4-digit year and they differ -> not similar
    y1, y2 = _extract_year(c1), _extract_year(c2)
    if y1 and y2 and y1 != y2:
        return False

    # Otherwise fall back to fuzzy string matcher (full string)
    return strings_similar(c1, c2, threshold=threshold, strip_common_winery_words=False)


# # Case 1: accents + spacing differences
# r1a = {"MakerName": "Château Margaux", "CustomID": "Château  Margaux|2019"}
# r1b = {"MakerName": "Chateau   Margaux", "CustomID": "Chateau Margaux | 2019"}

# # Case 2: extra word present (subset tokens)
# r2a = {"MakerName": "Robert Mondavi Winery", "CustomID": "Robert Mondavi Winery|2019"}
# r2b = {"MakerName": "Robert Mondavi",        "CustomID": "Robert Mondavi|2019"}

# # Case 3: punctuation/case differences
# r3a = {"MakerName": "Opus One", "CustomID": "Opus One|2019"}
# r3b = {"MakerName": "OPU-ON", "CustomID": "OPUS-ONE|2019"}

# print(isMakerNameSimilar(r1a, r1b))   # True
# print(isCustomIDSimilar(r1a, r1b))    # True

# print(isMakerNameSimilar(r2a, r2b))   # True
# print(isCustomIDSimilar(r2a, r2b))    # True

# print(isMakerNameSimilar(r3a, r3b))   # True
# print(isCustomIDSimilar(r3a, r3b))    # True


# # Case A
# rA1 = {"MakerName": "Opus One",          "CustomID": "Opus One|2018"}
# rA2 = {"MakerName": "Opus One",          "CustomID": "Opus One|2019"}
# print(isMakerNameSimilar(rA1, rA2))   # True
# print(isCustomIDSimilar(rA1, rA2))    # False

# # Case B
# rB1 = {"MakerName": "Chateau Margaux",   "CustomID": "Chateau Margaux|2015"}
# rB2 = {"MakerName": "Chateau Margaux",   "CustomID": "Chateau Margaux|2018"}
# print(isMakerNameSimilar(rB1, rB2))   # True
# print(isCustomIDSimilar(rB1, rB2))    # False

# # Case C
# rC1 = {"MakerName": "Robert Mondavi",    "CustomID": "Robert Mondavi|2007"}
# rC2 = {"MakerName": "Robert Mondavi",    "CustomID": "Robert Mondavi|2019"}
# print(isMakerNameSimilar(rC1, rC2))   # True
# print(isCustomIDSimilar(rC1, rC2))    # False
