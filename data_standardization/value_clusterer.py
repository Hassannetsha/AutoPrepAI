"""
value_clusterer.py
==================
Safe edit-distance graph clustering for categorical value standardization.

Design goals
------------
- Never merge values that differ in leading character unless they are
  single-character abbreviations (M -> Male) or pure case/punct variants.
- Preserve comparison operators (<= >) so "<=50K" and ">50K" stay separate.
- Classify each cluster as AUTO-RESOLVED or AMBIGUOUS so the LLM is only
  called on clusters that genuinely need semantic judgment.
- One LLM call per ambiguous cluster (not per value), drastically cutting
  token usage compared to per-value normalization.

Cluster modes
-------------
ALLOWED_ANCHOR   cluster has a canonical from the allowed_values whitelist;
                 all members are surface variants -> auto-resolve.
HEURISTIC_CANON  no allowed_values; canonical chosen by frequency + completeness;
                 pure case/punct cluster -> auto-resolve.
AMBIGUOUS        members cannot be auto-resolved; sent to LLM as a group.
SINGLETON        single value, no change needed (or orphan that failed to join
                 any cluster -> sent to LLM if not in allowed_values).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from rapidfuzz.distance import Levenshtein


# String normalisation helpers

def _strip(s: str) -> str:
    """
    Lowercase + unicode-normalise + remove punctuation EXCEPT comparison
    operators (<, >, =) so that '<=50K' and '>50K' remain distinct.
    """
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"[^a-z0-9<>=]+", "", s)


# Safe similarity

def safe_similar(a: str, b: str, threshold: float = 0.35) -> bool:
    """
    Return True only when *a* and *b* are likely surface variants of the
    same value (typo, case, punctuation, truncation/abbreviation).

    Conservative by design — it must NOT return True for semantically
    related but distinct values such as ('male', 'female').

    Rules applied in priority order
    --------------------------------
    1. Stripped exact match  (handles u.s. == us == US).
    2. One is a prefix of the other AND first chars match AND the prefix
       covers ≥50 % of the longer (or is a single-char abbreviation).
    3. Normalised Levenshtein distance ≤ threshold.
       - Single-char values: first char must match.
       - Multi-char values: if first chars differ, only allow distance == 1
         (a single-character transposition at the start).
    """
    a_s, b_s = _strip(a), _strip(b)
    if not a_s or not b_s:
        return False

    # Rule 1 — stripped exact match
    if a_s == b_s:
        return True

    shorter, longer = (a_s, b_s) if len(a_s) <= len(b_s) else (b_s, a_s)

    # Rule 2 — prefix / abbreviation
    if longer.startswith(shorter) and shorter[0] == longer[0]:
        if len(shorter) == 1 or len(shorter) / len(longer) >= 0.5:
            return True

    # Rule 3 — edit distance
    dist = Levenshtein.distance(a_s, b_s)
    norm = dist / max(len(a_s), len(b_s), 1)
    if norm > threshold:
        return False

    if len(a_s) == 1 or len(b_s) == 1:
        return a_s[0] == b_s[0]

    if a_s[0] != b_s[0] and dist > 1:
        return False

    return True


# Canonical-form heuristic (used when no allowed_values anchor exists)

def _completeness(v: str) -> int:
    """
    Score how 'complete' a value looks so we can prefer the canonical
    surface form over abbreviations or all-caps variants.

    Higher is better.
    """
    s = _strip(v)
    score = len(s)
    # All-caps short values are likely abbreviations — penalise them
    if v == v.upper() and len(v) <= 3:
        score -= 2
    # Mixed-case suggests a proper noun or formatted label
    if v != v.lower() and v != v.upper():
        score += 1
    return score


def _pick_canonical(members: list[str], value_counts: dict[str, int]) -> str:
    """
    Choose the canonical form for a cluster that has no allowed_values anchor.

    Priority: highest frequency → highest completeness score.
    """
    return max(members, key=lambda v: (value_counts.get(v, 0), _completeness(v)))


# Surface-variant detection

def _is_surface_variant(variant: str, canonical: str) -> bool:
    """
    Return True when *variant* differs from *canonical* only in case,
    punctuation, spacing, or is a recognised abbreviation of it.

    This is stricter than safe_similar — it excludes the edit-distance
    path so we can reliably separate 'pure formatting' from 'possible
    meaning difference'.
    """
    v_s, c_s = _strip(variant), _strip(canonical)

    # Exact stripped match
    if v_s == c_s:
        return True

    # Single-char abbreviation (M -> Male) if first chars match
    if len(v_s) == 1 and v_s[0] == c_s[0]:
        return True

    # Short prefix: e.g. 'mal' -> 'male'
    if c_s.startswith(v_s) and v_s[0] == c_s[0] and len(v_s) / max(len(c_s), 1) >= 0.5:
        return True

    return False


# Cluster dataclass

@dataclass
class Cluster:
    """
    One group of values that the similarity graph considers equivalent.

    Attributes
    ----------
    canonical   : str
        The chosen canonical form for this cluster.
    members     : list[str]
        All values in the cluster (including canonical itself).
    mode        : str
        'ALLOWED_ANCHOR'  – canonical is from allowed_values whitelist.
        'HEURISTIC_CANON' – canonical chosen by frequency/completeness.
        'SINGLETON'       – single value; if not in allowed, may be orphan.
    ambiguous   : bool
        True  → needs LLM to confirm / decide the mapping.
        False → auto-resolved; apply the mapping directly.
    reason      : str
        Human-readable explanation for paper / audit log.
    """
    canonical: str
    members: list[str]
    mode: str
    ambiguous: bool
    reason: str
    variants: list[str] = field(default_factory=list)   # members != canonical

    def mapping(self) -> dict[str, str]:
        """Return {variant: canonical} for all non-canonical members."""
        return {m: self.canonical for m in self.members if m != self.canonical}


# Main clustering function

def cluster_column(
    unique_values: list[str],
    value_counts: dict[str, int],
    allowed_values: Optional[list[str]] = None,
    similarity_threshold: float = 0.35,
) -> list[Cluster]:
    """
    Build a safe similarity graph over *unique_values* and return one
    :class:`Cluster` per connected component.

    Parameters
    ----------
    unique_values       : all unique non-null string values in the column.
    value_counts        : {value: row_count} for frequency-based tie-breaking.
    allowed_values      : optional whitelist of valid canonical forms.
    similarity_threshold: normalised edit-distance ceiling for an edge.

    Algorithm
    ---------
    With allowed_values
        Edges are only drawn from non-whitelisted values TO whitelisted values.
        This prevents non-allowed values from forming bridges between distinct
        canonical forms (e.g. 'mal' cannot connect 'Male' to 'Female').

    Without allowed_values
        Full pairwise edges among all unique values using safe_similar.
        Canonical is chosen by frequency + completeness heuristic.

    After component extraction, each cluster is classified:
    - SINGLETON     if it has exactly one member.
    - ALLOWED_ANCHOR if canonical is in allowed_values and all variants
                    are pure surface forms → ambiguous = False.
    - HEURISTIC_CANON if no allowed anchor but all members are pure surface
                    variants of the chosen canonical → ambiguous = False.
    - AMBIGUOUS     in all other cases → ambiguous = True.
    """
    allowed_set = set(allowed_values) if allowed_values else set()

    G = nx.Graph()
    G.add_nodes_from(unique_values)

    if allowed_set:
        # Directed-style: only link dirty value -> allowed value
        allowed_list = list(allowed_set)
        for val in unique_values:
            if val in allowed_set:
                continue
            for av in allowed_list:
                if safe_similar(val, av, threshold=similarity_threshold):
                    G.add_edge(val, av)
    else:
        # Full pairwise — O(n²) but n is the number of *unique* values,
        # which is small for categorical columns.
        for i, a in enumerate(unique_values):
            for b in unique_values[i + 1:]:
                if safe_similar(a, b, threshold=similarity_threshold):
                    G.add_edge(a, b)

    clusters: list[Cluster] = []

    for component in nx.connected_components(G):
        members = list(component)

        # --- Determine canonical and mode ---
        if allowed_set:
            allowed_in = [v for v in members if v in allowed_set]
            if allowed_in:
                canonical = allowed_in[0]   # there should be at most one per component
                mode = "ALLOWED_ANCHOR"
            else:
                # Orphan: didn't link to any allowed value
                canonical = _pick_canonical(members, value_counts)
                mode = "HEURISTIC_CANON"
        else:
            canonical = _pick_canonical(members, value_counts)
            mode = "HEURISTIC_CANON"

        variants = [m for m in members if m != canonical]

        # --- Singleton ---
        if len(members) == 1:
            # Singleton in an allowed context: if not in allowed_values, it's
            # an orphan that the LLM should handle.
            is_orphan = bool(allowed_set) and canonical not in allowed_set
            clusters.append(Cluster(
                canonical=canonical,
                members=members,
                mode="SINGLETON",
                ambiguous=is_orphan,
                reason=(
                    "orphan singleton: not in allowed_values"
                    if is_orphan else
                    "singleton: no change needed"
                ),
                variants=[],
            ))
            continue

        # --- Multi-member cluster: classify ambiguity ---
        all_surface = all(_is_surface_variant(v, canonical) for v in variants)

        if mode == "ALLOWED_ANCHOR" and all_surface:
            ambiguous = False
            reason = (
                f"allowed anchor '{canonical}'; all {len(variants)} variant(s) are "
                "pure case/punct/abbreviation forms → auto-resolved"
            )
        elif mode == "ALLOWED_ANCHOR" and not all_surface:
            ambiguous = True
            reason = (
                f"allowed anchor '{canonical}' but some variants are not pure surface "
                "forms → LLM needed to confirm mapping"
            )
        elif mode == "HEURISTIC_CANON" and all_surface:
            ambiguous = False
            reason = (
                f"heuristic canonical '{canonical}'; all variants are pure surface "
                "forms (case/punct) → auto-resolved"
            )
        else:
            ambiguous = True
            reason = (
                f"heuristic canonical '{canonical}' with non-surface variants → "
                "LLM needed to confirm grouping and canonical form"
            )

        clusters.append(Cluster(
            canonical=canonical,
            members=members,
            mode=mode,
            ambiguous=ambiguous,
            reason=reason,
            variants=variants,
        ))

    return clusters


# Convenience: split clusters into resolved vs ambiguous

def split_clusters(
    clusters: list[Cluster],
) -> tuple[list[Cluster], list[Cluster]]:
    """Return (auto_resolved, ambiguous) lists."""
    auto = [c for c in clusters if not c.ambiguous]
    amb = [c for c in clusters if c.ambiguous]
    return auto, amb