"""Lee skeleton, tip spur pruning, and the skeleton graph (Skan + networkx)."""
import networkx as nx
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

# Skan's branch_type codes; 3 is an isolated cycle (no tips)
_BRANCH_TYPE = {0: "tip-tip", 1: "tip-junction", 2: "junction-junction", 3: "tip-tip"}


def skeletonize_mask(mask):
    """Lee skeleton of a binary mask, as a boolean array."""
    return skeletonize(np.asarray(mask) > 0, method="lee") > 0


def prune_spurs(skel, min_spur=5):
    """Remove tip-junction spurs shorter than min_spur px; the junction pixel is kept."""
    out = skel.copy()
    if not skel.any():
        return out
    S = Skeleton(skel.astype(np.uint8))
    df = summarize(S, separator="_")
    for i, row in enumerate(df.itertuples(index=False)):
        if int(row.branch_type) == 1 and row.branch_distance < min_spur:
            idx = np.asarray(S.path(i))
            keep = S.degrees[idx] >= 3
            rc = S.coordinates[idx[~keep]].astype(int)
            out[rc[:, 0], rc[:, 1]] = False
    return out


def _edge_type(G, u, v):
    """Branch type from the degrees of its two end nodes."""
    ends = ("tip" if G.degree(n) == 1 else "junction" for n in (u, v))
    return "-".join(sorted(ends))


def _build_graph(skel):
    """Skan branches as a MultiGraph: nodes with 'pos', edges with 'length' and 'type'.

    A MultiGraph keeps both arcs of a small loop between the same two junctions."""
    G = nx.MultiGraph()
    if skel.sum() == 0:
        return G
    S = Skeleton(skel.astype(np.uint8))
    df = summarize(S, separator="_")
    for row in df.itertuples(index=False):
        src, dst = int(row.node_id_src), int(row.node_id_dst)
        for nid in (src, dst):
            if nid not in G:
                r, c = S.coordinates[nid]
                G.add_node(nid, pos=(float(r), float(c)))
        G.add_edge(src, dst, length=float(row.branch_distance), type=_BRANCH_TYPE.get(int(row.branch_type), "tip-tip"))
    return G


def _merge_degree2(G):
    """Join the two branches at every degree-2 node into one edge; real loops are kept."""
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if G.degree(n) != 2:
                continue
            edges = list(G.edges(n, keys=True, data=True))
            if len(edges) == 1:  # a self-loop counts as degree 2
                continue
            u = edges[0][1] if edges[0][0] == n else edges[0][0]
            v = edges[1][1] if edges[1][0] == n else edges[1][0]
            if u == v:  # both branches go to the same neighbour
                continue
            length = edges[0][3]["length"] + edges[1][3]["length"]
            G.remove_node(n)
            G.add_edge(u, v, length=length, type=_edge_type(G, u, v))
            changed = True
            break  # node set changed; restart the scan
    return G


def skeleton_to_graph(skel, min_spur=0):
    """Skeleton graph; with min_spur > 0, tip-junction branches shorter than min_spur are removed.

    Junction-junction branches are never removed, so no connectivity is cut."""
    G = _build_graph(skel)
    if min_spur <= 0 or G.number_of_edges() == 0:
        return G
    tips = []
    for u, v, d in G.edges(data=True):
        if d["type"] == "tip-junction" and d["length"] < min_spur:
            tip = u if G.degree(u) == 1 else (v if G.degree(v) == 1 else None)
            if tip is not None:
                tips.append(tip)
    G.remove_nodes_from(set(tips))
    return _merge_degree2(G)
