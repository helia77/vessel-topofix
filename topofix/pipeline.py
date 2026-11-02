"""Training-free topology repair: closing, cleaning, skeleton bridging, painting."""
import numpy as np
from skimage.morphology import closing, disk

from .bridge import clean_mask, find_path_bridges, paint_bridges
from .skeleton import prune_spurs, skeletonize_mask

MIN_AREA = 20   # px, components below this are removed after closing
MIN_SPUR = 5    # px, tip spurs below this are pruned from the skeleton

# method "same": targets on other components and on the tip's own piece (guard k); "path": other components only
LOOP_PAPER = dict(method="same", R=50, angle=20, p_min=0.5, k=8)
CROSS_COMP = dict(method="path", R=30, angle=20, p_min=0.2)


def bridges(skel, prob, setting):
    """Bridges of one setting on a pruned skeleton and the foreground probability map."""
    s = setting
    return find_path_bridges(skel, prob, s["R"], s["angle"], s["p_min"], allow_same=s["method"] == "same",
                             same_ratio=s.get("k", 2))


def refine(mask, prob, r, setting=LOOP_PAPER):
    """Closing with disk(r), cleaning, bridging on the pruned Lee skeleton, painting.

    Returns (refined mask, bridges). The paper uses r 2 (full data) and r 3 (single image).
    """
    clean = clean_mask(closing(np.asarray(mask) > 0, disk(r)), MIN_AREA)
    skel = prune_spurs(skeletonize_mask(clean), MIN_SPUR)
    bs = bridges(skel, prob, setting)
    return paint_bridges(clean, skel, bs), bs
