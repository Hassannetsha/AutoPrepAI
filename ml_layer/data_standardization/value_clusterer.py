
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

# Build an abbreviation from the first letters of each word in a string.
# (e.g. "United States" -> "us", "New York City" -> "nyc")
def _abbreviation(text):
    words = re.findall(r"[A-Za-z]+", text)
    return "".join(word[0] for word in words).lower()


# Safe similarity
def safe_similar(a: str, b: str, threshold: float = 0.35) -> bool:
    # checkis if two strings are similar enough to be considered equivalent, using a combination of rules and edit distance.
    
    # Stripped exact match  (handles u.s. == us == US).
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
        
    # Rule 2.5 — multi-word abbreviation
    if a_s == _abbreviation(b) or b_s == _abbreviation(a):
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


def _completeness(v: str) -> int:
    # how to choose the canonical form for a cluster
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
    Highest frequency → highest completeness score.
    """
    return max(members, key=lambda v: (value_counts.get(v, 0), _completeness(v)))


# Surface-variant detection
# checks if a variant is a surface-level variant of the canonical form
# meaning it can be automatically resolved without LLM intervention.
def _is_surface_variant(variant: str, canonical: str) -> bool:
    # decide to merge without asking the LLM 
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
    
    # Multi-word abbreviation
    if v_s == _abbreviation(canonical) or c_s == _abbreviation(variant):
        return True

    return False


# Cluster dataclass
@dataclass
class Cluster:
    """
    Represents a cluster of similar categorical values.
    """
    canonical: str # selected canonical value 
    members: list[str] # all values in the cluster, including canonical
    mode: str # allowed_anchor | heuristic_canon | singleton
    ambiguous: bool # true if LLM is needed
    reason: str # explanation of the clustering decision
    variants: list[str] = field(default_factory=list)   # members excluding the canonical

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
    # fast lookup for allowed canonical values
    allowed_set = set(allowed_values) if allowed_values else set()

    # build similarity graph
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
        # compare every pair of unique values when no allowed_values are provided
        for i, a in enumerate(unique_values):
            for b in unique_values[i + 1:]:
                if safe_similar(a, b, threshold=similarity_threshold):
                    G.add_edge(a, b)

    # store resulting cluster
    clusters: list[Cluster] = []

    for component in nx.connected_components(G):
        # process one cluster (connected component) at a time 
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

        # values to be replaced later (not canonical)
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


# Convenience: split clusters into auto-resolved vs ambiguous
def split_clusters(
    clusters: list[Cluster],
) -> tuple[list[Cluster], list[Cluster]]:
    """Return (auto_resolved, ambiguous) lists."""
    auto = [c for c in clusters if not c.ambiguous]
    amb = [c for c in clusters if c.ambiguous]
    return auto, amb