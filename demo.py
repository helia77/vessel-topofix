"""Synthetic ladder demo: cross-component vs loop bridging on a mask with broken rungs and a broken vessel.

Run from the repo root:  python demo.py [out.png]   (default demo.png)"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from skimage.filters import gaussian
from skimage.morphology import dilation, disk

from topofix import CROSS_COMP, LOOP_PAPER, refine
from topofix import metrics as m
from topofix.bridge import bridge_pixels


def ladder(h=200, w=256):
    """(gt, pred, prob): a ladder plus a separate vessel; pred has two broken rungs and a broken vessel."""
    S = np.zeros((h, w), bool)
    for r in (50, 130):
        S[line(r, 20, r, w - 21)] = True
    for c in (50, 100, 150, 200):
        S[line(50, c, 130, c)] = True
    S[line(150, 120, 185, 120)] = True
    gt = dilation(S, disk(3))
    pred = gt.copy()
    pred[85:97, 95:106] = False
    pred[85:97, 145:156] = False
    pred[165:177, 114:127] = False
    return gt, pred, np.clip(gaussian(gt.astype(float), 2), 0, 1)


def main(out="demo.png"):
    gt, pred, prob = ladder()
    rows = [("Prediction", pred, [])]
    rows += [(name, *refine(pred, prob, 2, st)) for name, st in (("Cross-component", CROSS_COMP), ("Loop bridging", LOOP_PAPER))]
    print(f"{'':<16}{'Dice':>7}{'clDice':>8}{'BIoU':>7}{'ASSD':>7}{'mu0':>5}{'mu1':>5}{'mu':>4}"
          f"{'b0 err':>8}{'b1 err':>8}{'ends':>6}{'junct':>7}{'wrong/bridges':>15}")
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground truth")
    for ax, (name, M, bs) in zip(axes[1:], rows):
        mu, be, (ends, jn) = m.betti_matching(M, gt), m.betti_error(M, gt), m.ends_junctions(M)
        ok = m.bridges_correct(bs, gt)
        print(f"{name:<16}{m.dice(M, gt):>7.3f}{m.cldice(M, gt):>8.3f}{m.boundary_iou(M, gt):>7.3f}{m.assd(M, gt):>7.3f}"
              f"{mu['mu0']:>5}{mu['mu1']:>5}{mu['mu']:>4}{be['b0_err']:>8}{be['b1_err']:>8}{ends:>6}{jn:>7}"
              f"{len(ok) - sum(ok):>9} / {len(ok)}")
        ax.imshow(M, cmap="gray")
        for b, good in zip(bs, ok):
            rr, cc = bridge_pixels(b)
            ax.plot(cc, rr, color="tab:green" if good else "tab:red", lw=2)
        ax.set_title(f"{name}  (mu {mu['mu']})")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}  (bridges: green correct, red wrong)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
