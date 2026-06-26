"""Build the substrate: a high-confidence human protein interaction network
and the set of genes known to be involved in Alzheimer disease.

Two public sources, no account, no API key:

* STRING v12.0 — the protein-protein interaction graph. We download the human
  links file once, keep edges scoring >= STRING_SCORE_MIN, and translate
  STRING's Ensembl protein ids to gene symbols.
* Open Targets — the Alzheimer "associated targets" list, queried live over
  its public GraphQL endpoint. Genes above SEED_SCORE_MIN become our seeds.

Everything is cached under data/ so the network is built once and reused.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

from . import config as C


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------
def _download(url: str, dest: Path) -> Path:
    """Stream a file to disk once; reuse it forever after."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(r.raw, fh)
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------
# STRING -> gene-symbol interaction graph
# --------------------------------------------------------------------------
def _load_protein_to_symbol() -> dict[str, str]:
    """STRING Ensembl-protein id -> preferred gene symbol."""
    info = _download(C.STRING_INFO_URL, C.DATA_DIR / "string_info.txt.gz")
    mapping: dict[str, str] = {}
    with gzip.open(info, "rt") as fh:
        next(fh)  # header
        for line in fh:
            protein_id, symbol = line.split("\t")[:2]
            mapping[protein_id] = symbol
    return mapping


def load_edges() -> pd.DataFrame:
    """High-confidence STRING edges as a gene-symbol pair table.

    Returns columns (gene_a, gene_b, score) with gene_a < gene_b, deduplicated.
    The raw file is ~11M edges; the score filter keeps the trustworthy core.
    """
    cache = C.DATA_DIR / "edges_highconf.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)

    links = _download(C.STRING_LINKS_URL, C.DATA_DIR / "string_links.txt.gz")
    sym = _load_protein_to_symbol()

    rows: list[tuple[str, str, int]] = []
    with gzip.open(links, "rt") as fh:
        next(fh)  # header: "protein1 protein2 combined_score"
        for line in fh:
            p1, p2, score = line.split()
            s = int(score)
            if s < C.STRING_SCORE_MIN:
                continue
            a, b = sym.get(p1), sym.get(p2)
            if a is None or b is None or a == b:
                continue
            if a > b:
                a, b = b, a
            rows.append((a, b, s))

    edges = (
        pd.DataFrame(rows, columns=["gene_a", "gene_b", "score"])
        .groupby(["gene_a", "gene_b"], as_index=False)["score"]
        .max()
    )
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_csv(cache, index=False, compression="gzip")
    return edges


# --------------------------------------------------------------------------
# Open Targets -> Alzheimer seed genes
# --------------------------------------------------------------------------
def load_seed_genes() -> pd.DataFrame:
    """Alzheimer-associated genes from Open Targets, above SEED_SCORE_MIN.

    Returns columns (symbol, score), sorted by score descending.
    """
    cache = C.DATA_DIR / "seeds_opentargets.csv"
    if cache.exists():
        return pd.read_csv(cache)

    query = """
    query ADTargets($efo: String!, $index: Int!) {
      disease(efoId: $efo) {
        associatedTargets(page: {index: $index, size: 100}) {
          count
          rows { target { approvedSymbol } score }
        }
      }
    }"""
    symbols: list[str] = []
    scores: list[float] = []
    index = 0
    done = False
    while not done:
        resp = requests.post(
            C.OPENTARGETS_GRAPHQL,
            json={"query": query, "variables": {"efo": C.DISEASE_ID, "index": index}},
            timeout=60,
        )
        resp.raise_for_status()
        block = resp.json()["data"]["disease"]["associatedTargets"]
        for row in block["rows"]:
            # rows come back sorted by score descending, so the first one below
            # the bar means every remaining gene is below it too.
            if row["score"] < C.SEED_SCORE_MIN:
                done = True
                break
            symbols.append(row["target"]["approvedSymbol"])
            scores.append(row["score"])
        index += 1
        if index * 100 >= block["count"]:
            break

    seeds = (
        pd.DataFrame({"symbol": symbols, "score": scores})
        .query("score >= @C.SEED_SCORE_MIN")
        .drop_duplicates("symbol")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(cache, index=False)
    return seeds


# --------------------------------------------------------------------------
# assembled graph
# --------------------------------------------------------------------------
class Network:
    """Immutable view of the interactome plus the seed labels.

    nodes      : list[str]        gene symbols, index order matches the matrix
    index      : dict[str, int]   symbol -> row
    adjacency  : csr_matrix       symmetric, weighted by STRING score/1000
    seed_mask  : np.ndarray bool  True where the gene is a known AD gene
    """

    def __init__(self, nodes, index, adjacency, seed_mask):
        self.nodes = nodes
        self.index = index
        self.adjacency = adjacency
        self.seed_mask = seed_mask

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def n_seeds(self) -> int:
        return int(self.seed_mask.sum())


def build_network() -> Network:
    """Edges + seeds -> a single connected graph restricted to the largest
    component, with seeds that survived the mapping marked."""
    edges = load_edges()
    seeds = load_seed_genes()

    nodes = sorted(set(edges["gene_a"]) | set(edges["gene_b"]))
    index = {g: i for i, g in enumerate(nodes)}
    n = len(nodes)

    rows = edges["gene_a"].map(index).to_numpy()
    cols = edges["gene_b"].map(index).to_numpy()
    weights = edges["score"].to_numpy(dtype=float) / 1000.0
    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    adj = adj + adj.T  # symmetric

    seed_mask = np.zeros(n, dtype=bool)
    for sym in seeds["symbol"]:
        i = index.get(sym)
        if i is not None:
            seed_mask[i] = True

    net = Network(nodes, index, adj.tocsr(), seed_mask)
    return _largest_component(net)


def _largest_component(net: Network) -> Network:
    """Keep the giant connected component — disease propagation is only
    meaningful inside one reachable graph."""
    n_comp, labels = sp.csgraph.connected_components(net.adjacency, directed=False)
    if n_comp == 1:
        return net
    biggest = np.bincount(labels).argmax()
    keep = np.where(labels == biggest)[0]
    nodes = [net.nodes[i] for i in keep]
    index = {g: i for i, g in enumerate(nodes)}
    adj = net.adjacency[keep][:, keep]
    seed_mask = net.seed_mask[keep]
    return Network(nodes, index, adj.tocsr(), seed_mask)


def summary() -> dict:
    """Small JSON-able description of the assembled graph (used in results)."""
    net = build_network()
    deg = np.asarray((net.adjacency > 0).sum(axis=1)).ravel()
    return {
        "disease": C.DISEASE_NAME,
        "nodes": net.n,
        "edges": int((net.adjacency > 0).nnz // 2),
        "seed_genes": net.n_seeds,
        "median_degree": int(np.median(deg)),
        "string_score_min": C.STRING_SCORE_MIN,
        "seed_score_min": C.SEED_SCORE_MIN,
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
