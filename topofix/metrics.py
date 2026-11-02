"""Pixel, topology and bridge metrics of a binary prediction against a binary ground truth."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.graph import MCP_Geometric
from skimage.measure import label

from .skeleton import skeleton_to_graph, skeletonize_mask

_BETTI_FILE = Path(__file__).resolve().parents[1] / "third_party" / "BettiMatching.py"
_BM = None


def dice(pred, gt):
    """Dice overlap of two binary masks."""
    pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
    return float(2 * (pred & gt).sum() / (pred.sum() + gt.sum()))


def cldice(pred, gt):
    """clDice (Shit et al., CVPR 2021) on Lee skeletons of both masks."""
    pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
    sp, sg = skeletonize_mask(pred), skeletonize_mask(gt)
    tprec = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    tsens = (sg & pred).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tprec * tsens / (tprec + tsens)) if tprec + tsens else 0.0


def boundary_iou(pred, gt, d=2):
    """Boundary IoU (Cheng et al., CVPR 2021): band = mask minus mask eroded d times by a 3 x 3 square.

    The crop edge does not count as boundary. Returns 1.0 if both bands are empty.
    """
    band = lambda M: (M > 0) & ~binary_erosion(M > 0, np.ones((3, 3)), iterations=d, border_value=1)
    P, G = band(np.asarray(pred)), band(np.asarray(gt))
    union = (P | G).sum()
    return float((P & G).sum() / union) if union else 1.0


def _contour(M):
    """Mask pixels within 1 px of the background (hole edges included, crop edge not)."""
    return M & (distance_transform_edt(M) <= 1) if not M.all() else np.zeros_like(M)


def assd(pred, gt, px=1.0):
    """Average symmetric surface distance between the two contours, times the pixel size px.

    nan if either contour is empty.
    """
    P, G = _contour(np.asarray(pred) > 0), _contour(np.asarray(gt) > 0)
    if not (P.any() and G.any()):
        return float("nan")
    d = np.concatenate([distance_transform_edt(~G)[P], distance_transform_edt(~P)[G]])
    return float(d.mean() * px)


def betti_numbers(M):
    """(b0, b1): 8-connected foreground components, 4-connected background holes off the border."""
    M = np.asarray(M) > 0
    bg = label(~M, connectivity=1)
    border = np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    return int(label(M, connectivity=2).max()), len(np.setdiff1d(np.unique(bg), np.append(border, 0)))


def betti_error(pred, gt):
    """Absolute Betti number errors |b0 - b0_gt| and |b1 - b1_gt|."""
    (p0, p1), (g0, g1) = betti_numbers(pred), betti_numbers(gt)
    return dict(b0_err=abs(p0 - g0), b1_err=abs(p1 - g1))


def _betti_module():
    """third_party/BettiMatching.py, loaded once; its plotting imports are stubbed, np.infty restored."""
    global _BM
    if _BM is None:
        np.infty = np.inf  # removed in numpy 2, still used by the third-party code
        pn, pi, gd = (types.ModuleType(n) for n in ("panel", "panel.interact", "gudhi"))
        pn.extension, pn.widgets, pi.interact, pi.fixed = (lambda *a, **k: None), types.ModuleType("panel.widgets"), None, None
        gd.wasserstein = types.ModuleType("gudhi.wasserstein")
        for name, mod in (("panel", pn), ("panel.interact", pi), ("panel.widgets", pn.widgets),
                          ("gudhi", gd), ("gudhi.wasserstein", gd.wasserstein)):
            sys.modules.setdefault(name, mod)
        spec = importlib.util.spec_from_file_location("BettiMatching", _BETTI_FILE)
        _BM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BM)
    return _BM


def betti_matching(pred, gt):
    """Betti matching error (Stucki et al., ICML 2023): unmatched features per dimension.

    Superlevel filtration, T-construction (foreground 8-, background 4-connected), non-relative.
    """
    R = _betti_module().BettiMatching(np.asarray(pred, float), np.asarray(gt, float), relative=False,
                                      filtration="superlevel", construction="T")
    extra = [len(R.unmatched_0[d]) for d in (0, 1)]
    missing = [len(R.unmatched_1[d]) for d in (0, 1)]
    return dict(mu0=extra[0] + missing[0], mu1=extra[1] + missing[1], mu=sum(extra) + sum(missing),
                extra0=extra[0], missing0=missing[0], extra1=extra[1], missing1=missing[1])


def ends_junctions(M, min_spur=5):
    """(ends, junctions): degree-1 and degree >= 3 nodes of the Lee skeleton graph, spurs < min_spur pruned."""
    G = skeleton_to_graph(skeletonize_mask(M), min_spur=min_spur)
    deg = [d for _, d in G.degree()]
    return sum(d == 1 for d in deg), sum(d >= 3 for d in deg)


def bridges_correct(bridges, gt, tol=2):
    """Per bridge, True if both ends are within tol px of one GT component and the path between

    them inside the GT mask is at most 2 x the bridge length (no shortcut between two branches).
    """
    gt = np.asarray(gt) > 0
    lab = label(gt, connectivity=2)
    d, (ir, ic) = distance_transform_edt(~gt, return_indices=True)
    cost = np.where(gt, 1.0, np.inf)
    snap = lambda p: (ir[int(p[0]), int(p[1])], ic[int(p[0]), int(p[1])])
    piece = lambda p: lab[snap(p)] if d[int(p[0]), int(p[1])] <= tol else 0
    out = []
    for b in bridges:
        ok = bool(piece(b[0]) and piece(b[0]) == piece(b[-1]))
        if ok:
            L = np.linalg.norm(np.diff(np.asarray(b, dtype=float), axis=0), axis=1).sum()
            acc, _ = MCP_Geometric(cost).find_costs([snap(b[0])], [snap(b[-1])])
            ok = bool(acc[snap(b[-1])] <= 2 * L)
        out.append(ok)
    return out
