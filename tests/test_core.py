"""Core tests on synthetic masks. Run:  python -m pytest tests  or  python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np
from skimage.draw import line
from skimage.morphology import closing, dilation, disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import ladder
from topofix import CROSS_COMP, LOOP_PAPER, refine
from topofix import metrics as m
from topofix.bridge import clean_mask
from topofix.skeleton import prune_spurs


def ring(n=64, r_out=20, r_in=14):
    yy, xx = np.mgrid[:n, :n]
    d = np.hypot(yy - n / 2, xx - n / 2)
    return (d <= r_out) & (d > r_in)


def test_clean_mask_drops_small_components():
    M = np.zeros((40, 40), bool)
    M[5:15, 5:15] = True
    M[30:33, 30:33] = True
    assert np.array_equal(clean_mask(M, 20), M & (np.arange(40)[:, None] < 20))


def test_prune_spurs_removes_short_spur_only():
    S = np.zeros((60, 60), bool)
    S[line(30, 5, 30, 54)] = True
    S[line(27, 30, 29, 30)] = True  # 3 px spur
    S[line(31, 20, 50, 20)] = True  # 20 px branch
    P = prune_spurs(S, 5)
    assert not P[27:30, 30].any() and P[31:51, 20].all() and P[30, 5:55].all()


def test_refine_without_gaps_adds_nothing():
    gt, _, prob = ladder()
    out, bs = refine(gt, prob, 2, LOOP_PAPER)
    assert bs == [] and np.array_equal(out, clean_mask(closing(gt, disk(2)), 20))


def test_loop_bridging_closes_broken_rungs():
    gt, pred, prob = ladder()
    out, bs = refine(pred, prob, 2, LOOP_PAPER)
    assert len(bs) == 3 and all(m.bridges_correct(bs, gt))
    assert m.betti_matching(out, gt)["mu"] == 0 and m.betti_numbers(out) == m.betti_numbers(gt)


def test_cross_component_cannot_close_a_loop():
    gt, pred, prob = ladder()
    out, _ = refine(pred, prob, 2, CROSS_COMP)
    assert m.betti_matching(out, gt)["missing1"] == 2


def test_metrics_identity():
    gt = ladder()[0]
    assert m.dice(gt, gt) == 1 and m.cldice(gt, gt) == 1 and m.boundary_iou(gt, gt) == 1 and m.assd(gt, gt) == 0
    assert m.betti_matching(gt, gt)["mu"] == 0 and m.betti_error(gt, gt) == dict(b0_err=0, b1_err=0)


def test_ring_vs_disk_misses_one_loop():
    R = ring()
    D = ring(r_in=-1)  # filled disk
    assert m.betti_numbers(R) == (1, 1) and m.betti_numbers(D) == (1, 0)
    mu = m.betti_matching(D, R)
    assert (mu["mu0"], mu["missing1"], mu["extra1"]) == (0, 1, 0)


def test_ends_junctions_of_a_t():
    M = np.zeros((80, 80), bool)
    M[line(20, 10, 20, 70)] = True
    M[line(20, 40, 70, 40)] = True
    assert m.ends_junctions(dilation(M, disk(2))) == (3, 1)


def test_bridges_correct():
    gt = np.zeros((60, 60), bool)
    gt[10:14, 5:55] = True
    gt[40:44, 5:55] = True
    along = [(12, c) for c in range(10, 31)]
    across = [(r, 30) for r in range(12, 43)]
    assert m.bridges_correct([along, across], gt) == [True, False]


if __name__ == "__main__":
    tests = [f for k, f in sorted(globals().items()) if k.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
