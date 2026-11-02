"""Minimal-path bridges from broken skeleton tips, and painting them into the mask."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skan import Skeleton
from skimage.graph import MCP_Geometric, route_through_array
from skimage.measure import label

from .skeleton import skeleton_to_graph

MIN_SPUR = 5      # tips come from skeleton_to_graph(skel, min_spur=5)
SAME_ADD = 30     # own-piece guard: along-skeleton >= max(k x straight, straight + 30)
MARGIN = 10       # routing window margin around tip and target, px
END = 8           # px around a bridge end where the vessel radius is read


def clean_mask(mask, min_area):
    """Remove 8-connected components smaller than min_area pixels."""
    m = np.asarray(mask) > 0
    lab = label(m, connectivity=2)
    out = np.zeros_like(m)
    for i in range(1, lab.max() + 1):
        comp = lab == i
        if comp.sum() >= min_area:
            out |= comp
    return out


def _tip_pixels(skel, tip, n):
    """The tip plus up to n skeleton pixels walked greedily inward."""
    h, w = skel.shape
    visited, path, cur = {tip}, [tip], tip
    while len(path) < n:
        r, c = cur
        nxt = next(((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w
                    and skel[r + dr, c + dc] and (r + dr, c + dc) not in visited), None)
        if nxt is None:
            break
        visited.add(nxt)
        path.append(nxt)
        cur = nxt
    return np.array(path, dtype=float)


def _tip_branches(skel):
    """{tip pixel: its own branch (row, col), tip first, cut before the first junction}."""
    S = Skeleton(skel.astype(np.uint8))
    out = {}
    for i in range(S.n_paths):
        idx = np.asarray(S.path(i))
        pts, deg = S.coordinates[idx].astype(int), S.degrees[idx]
        for o in (slice(None), slice(None, None, -1)):  # from either end
            if deg[o][0] == 1:
                keep = np.cumprod(deg[o] < 3).astype(bool)
                out[tuple(pts[o][0])] = pts[o][keep].astype(float)
    return out


def _direction(pts, n=8):
    """Outward unit vector: tip minus the mean of the first n pixels."""
    pts = pts[:n]
    if len(pts) < 2:
        return np.zeros(2)
    v = pts[0] - pts.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm else np.zeros(2)


def _along_skeleton(skel, start):
    """Path length along the skeleton from start to every pixel; inf elsewhere."""
    acc, _ = MCP_Geometric(np.where(skel, 1.0, np.inf)).find_costs([start])
    return acc


def _tip_candidates(skel, R, max_angle, border_margin, allow_same, same_ratio):
    """Yield (tip, {component: (dist, nearest pixel)}) for skeleton pixels within R px and the angle cone.

    Own-component pixels count only with allow_same, and only if far enough along the skeleton.
    """
    lab = label(skel, connectivity=2)
    G = skeleton_to_graph(skel, min_spur=MIN_SPUR)
    endpoints = [d["pos"] for n, d in G.nodes(data=True) if G.degree(n) == 1]
    branches = _tip_branches(skel)
    comp_ids = range(1, lab.max() + 1)
    coords = {cid: np.argwhere(lab == cid).astype(float) for cid in comp_ids}
    trees = {cid: cKDTree(coords[cid]) for cid in comp_ids}
    h, w = skel.shape
    cos_min = np.cos(np.deg2rad(max_angle))
    for pos in endpoints:
        er, ec = int(round(pos[0])), int(round(pos[1]))
        if min(er, ec, h - 1 - er, w - 1 - ec) < border_margin:  # tip at the image edge
            continue
        tip_cid = lab[er, ec]
        if tip_cid == 0:  # snap an off-pixel node position to the nearest skeleton pixel
            nz = np.argwhere(skel).astype(float)
            j = np.argmin(((nz - [pos[0], pos[1]]) ** 2).sum(axis=1))
            er, ec = int(nz[j, 0]), int(nz[j, 1])
            tip_cid = lab[er, ec]
        pts = branches.get((er, ec))
        direction = _direction(_tip_pixels(skel, (er, ec), 8) if pts is None else pts)
        if np.linalg.norm(direction) == 0:
            continue
        if allow_same:
            along = _along_skeleton(skel, (er, ec))
        cands = {}
        for cid in comp_ids:
            if cid == tip_cid and not allow_same:
                continue
            for k in trees[cid].query_ball_point([er, ec], R):
                tr, tc = coords[cid][k]
                to = np.array([tr - er, tc - ec])
                dist = np.linalg.norm(to)
                if dist == 0 or np.dot(direction, to / dist) < cos_min:
                    continue
                if cid == tip_cid and along[int(tr), int(tc)] < max(same_ratio * dist, dist + SAME_ADD):
                    continue
                if cid not in cands or dist < cands[cid][0]:
                    cands[cid] = (dist, (int(round(tr)), int(round(tc))))
        yield (er, ec), cands


def find_path_bridges(skel, prob, R, max_angle, p_min, allow_same=False, same_ratio=2, border_margin=3):
    """Per tip, minimal paths on 1 - prob to each candidate; the cheapest with mean prob >= p_min is kept.

    Returns a list of paths, each a list of (row, col) from tip to target; A->B and B->A kept once.
    """
    skel = np.asarray(skel) > 0
    if not skel.any():
        return []
    prob = np.asarray(prob, dtype=float)
    cost = 1.0 - prob + 0.01
    h, w = cost.shape
    out, seen = [], set()
    for (er, ec), cands in _tip_candidates(skel, R, max_angle, border_margin, allow_same, same_ratio):
        routes = []
        for _, (tr, tc) in cands.values():
            r0, r1 = max(0, min(er, tr) - MARGIN), min(h, max(er, tr) + MARGIN + 1)
            c0, c1 = max(0, min(ec, tc) - MARGIN), min(w, max(ec, tc) + MARGIN + 1)
            local, _ = route_through_array(cost[r0:r1, c0:c1], (er - r0, ec - c0), (tr - r0, tc - c0),
                                           fully_connected=True, geometric=True)
            path = [(r + r0, c + c0) for r, c in local]
            routes.append((float(np.mean([cost[p] for p in path])), path))
        for _, path in sorted(routes, key=lambda mp: mp[0]):  # stable: ties keep candidate order
            pts = np.asarray(path)
            if prob[pts[:, 0], pts[:, 1]].mean() >= p_min:
                key = tuple(sorted((tuple(path[0]), tuple(path[-1]))))
                if key not in seen:
                    seen.add(key)
                    out.append(path)
                break
    return out


def bridge_pixels(bridge):
    """(rr, cc) pixels of a bridge path."""
    pts = np.asarray(bridge, dtype=int)
    return pts[:, 0], pts[:, 1]


def _bridge_radius(skel, bridges, dt):
    """Radius on bridge pixels: per end the median dt over skeleton px within END px, linear in between."""
    sr, sc = np.nonzero(skel)
    rad = np.zeros(dt.shape)

    def end_r(r, c):
        near = np.hypot(sr - r, sc - c) <= END
        return float(np.median(dt[sr[near], sc[near]])) if near.any() else 1.0

    for b in bridges:
        rr, cc = bridge_pixels(b)
        t = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rr), np.diff(cc)))])
        t = t / t[-1] if t[-1] > 0 else t
        rad[rr, cc] = np.maximum(rad[rr, cc], (1 - t) * end_r(rr[0], cc[0]) + t * end_r(rr[-1], cc[-1]))
    return rad


def _disc_union(rad):
    """Union of discs |x - p| <= rad[p] over every pixel p with rad[p] > 0."""
    K = int(np.ceil(rad.max()))
    h, w = rad.shape
    out = np.zeros((h + 2 * K, w + 2 * K), bool)
    yy, xx = np.mgrid[-K:K + 1, -K:K + 1]
    d2 = yy ** 2 + xx ** 2
    for r, c in np.argwhere(rad > 0):
        out[r:r + 2 * K + 1, c:c + 2 * K + 1] |= d2 <= rad[r, c] ** 2
    return out[K:K + h, K:K + w]


def paint_bridges(mask, skel, bridges, scale=0.8):
    """Mask plus discs of radius scale x local vessel radius (at least 1 px) on every bridge pixel."""
    rb = _bridge_radius(skel, bridges, distance_transform_edt(mask))
    return mask | _disc_union(np.where(rb > 0, np.maximum(scale * rb, 1.0), 0.0))
