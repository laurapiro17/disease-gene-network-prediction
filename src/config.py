"""Central knobs. Everything the pipeline reads lives here, so a reviewer can
see — and a test can override — every choice in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

# --- Disease ------------------------------------------------------------
# Alzheimer disease in the EFO/MONDO ontology used by Open Targets.
DISEASE_ID = "MONDO_0004975"
DISEASE_NAME = "Alzheimer disease"

# --- STRING interactome -------------------------------------------------
# Human (taxon 9606), STRING v12.0. Files stream in on first run and are
# cached under data/. We keep only high-confidence links: STRING's combined
# score runs 0-1000; 700 is its own "high confidence" cutoff.
STRING_VERSION = "12.0"
STRING_TAXON = "9606"
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)
STRING_SCORE_MIN = 700

# --- Alzheimer seed genes (Open Targets) --------------------------------
# A gene is treated as a "known" Alzheimer gene when its overall
# target-disease association score clears this bar. The score blends genetic,
# literature and pathway evidence; 0.5 is deliberately strict so the seeds are
# defensible, not a long tail of weak hits.
OPENTARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
SEED_SCORE_MIN = 0.5

# --- Evaluation ---------------------------------------------------------
# We hide a fraction of seed genes, prioritise the whole genome from the rest,
# and check whether the hidden ones surface. Same splits feed every model.
N_SPLITS = 5
RANDOM_STATE = 17

# --- Models -------------------------------------------------------------
RWR_RESTART = 0.4          # random-walk-with-restart restart probability
EMBED_DIM = 128            # spectral embedding dimensionality
CONFORMAL_ALPHA = 0.1      # 1 - target coverage for the candidate set
