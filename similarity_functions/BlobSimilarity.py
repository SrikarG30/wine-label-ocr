import math
import numpy as np

# --------- utilities (unchanged) ---------
def _roi_hw(blobdata):
    if "roi_shape" in blobdata and len(blobdata["roi_shape"]) == 2:
        H, W = int(blobdata["roi_shape"][0]), int(blobdata["roi_shape"][1])
        return max(1, H), max(1, W)
    if "roi_bbox" in blobdata and len(blobdata["roi_bbox"]) == 4:
        x0, y0, x1, y1 = blobdata["roi_bbox"]
        return max(1, int(y1 - y0)), max(1, int(x1 - x0))
    xs, ys, ws, hs = [], [], [], []
    for b in blobdata.get("blobs", []):
        xs.append(b.get("x", 0)); ys.append(b.get("y", 0))
        ws.append(b.get("w", 0)); hs.append(b.get("h", 0))
    H = int(max(1, max((y + h) for y, h in zip(ys or [1], hs or [1]))))
    W = int(max(1, max((x + w) for x, w in zip(xs or [1], ws or [1]))))
    return H, W

def _extract_features(blobdata):
    H, W = _roi_hw(blobdata)
    A_roi = float(H * W)
    feats = {"pos": [], "scale": [], "ratio": [], "sol": [], "ext": []}
    for b in blobdata.get("blobs", []):
        x = float(b.get("x", 0.0)); y = float(b.get("y", 0.0))
        w = float(b.get("w", 0.0)); h = float(b.get("h", 0.0))
        cx = float(b.get("cx", x + w * 0.5)); cy = float(b.get("cy", y + h * 0.5))
        area = float(b.get("area", max(1.0, w * h * 0.5)))
        ratio = float(np.clip(w / max(h, 1e-6), 1e-3, 1e3))
        extent = b.get("extent", area / max(w * h, 1e-6))
        solidity = b.get("solidity", 1.0)
        feats["pos"].append([cx / W, cy / H])
        feats["scale"].append(math.sqrt(max(area, 1.0)) / math.sqrt(A_roi))
        feats["ratio"].append(ratio)
        feats["sol"].append(float(np.clip(solidity, 0.0, 1.0)))
        feats["ext"].append(float(np.clip(extent,   0.0, 1.0)))
    for k in feats: feats[k] = np.asarray(feats[k], dtype=np.float32)
    return feats, (H, W)

def _pairwise_d2(A, B):
    a2 = np.sum(A*A, axis=1, keepdims=True)
    b2 = np.sum(B*B, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (A @ B.T)

def _estimate_translation(posA, posB):
    if len(posA) == 0 or len(posB) == 0:
        return np.zeros(2, np.float32)
    d2 = _pairwise_d2(posA, posB)
    j = np.argmin(d2, axis=1)
    return np.median(posA - posB[j], axis=0).astype(np.float32)

def _build_cost_matrix(Fa, Fb, shift, weights):
    pa = Fa["pos"]; pb = Fb["pos"] + shift[None, :]
    dpos = np.sqrt(np.maximum(_pairwise_d2(pa, pb), 0.0))
    def expand(a, b): return a[:, None], b[None, :]
    sa, sb = expand(Fa["scale"], Fb["scale"])
    ra, rb = expand(Fa["ratio"], Fb["ratio"])
    sola, solb = expand(Fa["sol"], Fb["sol"])
    exta, extb = expand(Fa["ext"], Fb["ext"])
    dscale = np.abs(sa - sb)
    dratio = np.abs(np.log(ra / np.maximum(rb, 1e-8)))
    dsol   = np.abs(sola - solb)
    dext   = np.abs(exta - extb)
    w_pos, w_s, w_r, w_sol, w_ext = (
        weights.get("pos", 0.60),
        weights.get("scale", 0.15),
        weights.get("ratio", 0.10),
        weights.get("solidity", 0.075),
        weights.get("extent", 0.075),
    )
    return (w_pos * dpos +
            w_s   * dscale +
            w_r   * np.minimum(dratio, 2.0) +
            w_sol * dsol +
            w_ext * dext)

def _assign(cost, pair_threshold=0.22):
    na, nb = cost.shape
    if na == 0 or nb == 0: return []
    try:
        from scipy.optimize import linear_sum_assignment
        gated = cost.copy(); gated[gated > pair_threshold] = 1e6
        ia, ib = linear_sum_assignment(gated)
        return [(a, b, float(cost[a, b])) for a, b in zip(ia, ib) if cost[a, b] <= pair_threshold]
    except Exception:
        idx = np.argsort(cost, axis=None); used_a = np.zeros(na, bool); used_b = np.zeros(nb, bool)
        pairs = []
        for flat in idx:
            a, b = divmod(flat, nb); c = cost[a, b]
            if c > pair_threshold: break
            if not used_a[a] and not used_b[b]:
                used_a[a] = used_b[b] = True; pairs.append((a, b, float(c)))
        return pairs

def _score(nA, nB, pairs, costs, pair_threshold):
    if nA == 0 and nB == 0: return 1.0, 1.0, 0.0
    if nA == 0 or nB == 0 or len(pairs) == 0: return 0.0, 0.0, 1.0
    coverage = len(pairs) / float(max(nA, nB))
    med_cost = float(np.median(costs)) if costs else pair_threshold
    quality = max(0.0, 1.0 - (med_cost / max(pair_threshold, 1e-6)))
    score = 2 * (coverage * quality) / max(coverage + quality, 1e-6)  # harmonic-like
    return float(np.clip(score, 0.0, 1.0)), coverage, med_cost

# --------- public API with tunable thresholds ---------
def isBlobDataSimilar(record1, record2, *,
                      threshold=0.55,            # final decision threshold on [0,1] score
                      pair_threshold=0.22,       # max allowed per-pair cost (gate)
                      weights=None,              # optional dict to tweak component weights
                      return_details=False):
    """
    Compare record1['BlobData'] and record2['BlobData'].

    threshold:      decide 'similar' if similarity_score >= threshold
    pair_threshold: gate for counting a blob pair as a match (lower = stricter)
    weights:        e.g. {'pos':0.6,'scale':0.15,'ratio':0.1,'solidity':0.075,'extent':0.075}
    """
    weights = weights or {}
    B1 = (record1 or {}).get("BlobData", {}) or {}
    B2 = (record2 or {}).get("BlobData", {}) or {}
    F1, _ = _extract_features(B1)
    F2, _ = _extract_features(B2)
    n1, n2 = len(F1["pos"]), len(F2["pos"])
    if n1 == 0 or n2 == 0:
        details = {'score': 0.0, 'coverage': 0.0, 'median_pair_cost': 1.0,
                   'matched_pairs': 0, 'n1': n1, 'n2': n2,
                   'pair_threshold': pair_threshold, 'decision_threshold': threshold}
        return (False, details) if return_details else False

    shift = _estimate_translation(F1["pos"], F2["pos"])
    cost = _build_cost_matrix(F1, F2, shift, weights)
    pairs = _assign(cost, pair_threshold=pair_threshold)
    costs = [c for *_, c in pairs]
    score, coverage, med_cost = _score(n1, n2, pairs, costs, pair_threshold)
    similar = score >= threshold
    details = {
        'score': score,
        'coverage': coverage,
        'median_pair_cost': med_cost,
        'matched_pairs': len(pairs),
        'n1': n1, 'n2': n2,
        'pair_threshold': pair_threshold,
        'decision_threshold': threshold
    }
    return (similar, details) if return_details else similar




# Paste blob data dict for testing
blobdata1 = {}
blobdata2 = {}


# ----- For Testing Blob Similarity ----- #
record1 = {"BlobData": blobdata1}
record2 = {"BlobData": blobdata2}

#   Run test (tune thresholds as needed)  #
similar, info = isBlobDataSimilar(
    record1, record2,
    threshold=0.55,        # final decision cutoff on the 0..1 score
    pair_threshold=0.22,   # per-pair gating cost (lower = stricter)
    return_details=True
)

print("similar?", similar)
print(info)


