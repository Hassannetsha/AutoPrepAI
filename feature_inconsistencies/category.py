"""
FIXED VERSION: Addresses the issues in your output
- Better high cardinality detection
- Proper case normalization before similarity detection
- Better spelling detection
- Negation pair detection (active/inactive)
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter
import re


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CategoryIssue:
    """Represents a detected issue in categorical data."""
    issue_type: str  
    affected_values: List[str]
    suggested_resolution: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[Dict] = None  # For extra info


@dataclass
class ResolutionMapping:
    """Maps original values to their resolved values."""
    original: str
    resolved: str
    reason: str


# ============================================================================
# IMPROVED DETECTORS
# ============================================================================

class HighCardinalityDetector:
    """Detects if a column is too unique to be truly categorical."""
    
    def __init__(self, uniqueness_threshold: float = 0.95, min_samples: int = 10):
        """
        Args:
            uniqueness_threshold: If unique_values/total_values > this, flag it
            min_samples: Minimum samples needed to make a determination
        """
        self.uniqueness_threshold = uniqueness_threshold
        self.min_samples = min_samples
    
    def detect(self, values: List[str]) -> Optional[CategoryIssue]:
        """Check if column has too many unique values."""
        if not values or len(values) < self.min_samples:
            return None
        
        unique_count = len(set(values))
        total_count = len(values)
        uniqueness_ratio = unique_count / total_count
        
        if uniqueness_ratio > self.uniqueness_threshold:
            return CategoryIssue(
                issue_type='high_cardinality',
                affected_values=[],
                suggested_resolution=f"Column may not be categorical (uniqueness: {uniqueness_ratio:.2%})",
                confidence=uniqueness_ratio,
                metadata={'unique_count': unique_count, 'total_count': total_count}
            )
        return None


class PlaceholderDetector:
    """Detects placeholder values like 'Unknown', 'Error', 'N/A', etc."""
    
    def __init__(self, custom_placeholders: Optional[Set[str]] = None):
        """
        Args:
            custom_placeholders: Additional placeholder values to detect
        """
        self.placeholders = {
            'unknown', 'n/a', 'na', 'null', 'none', 'error', 
            'missing', 'undefined', 'blank', 'empty', '-', '', 'n.a.',
            'not available', 'not applicable', 'nil'
        }
        if custom_placeholders:
            self.placeholders.update(p.lower().strip() for p in custom_placeholders)
    
    def detect(self, values: List[str]) -> Optional[CategoryIssue]:
        """Find placeholder values in the data."""
        found_placeholders = [
            v for v in set(values) 
            if v.lower().strip() in self.placeholders
        ]
        
        if found_placeholders:
            return CategoryIssue(
                issue_type='placeholder',
                affected_values=found_placeholders,
                suggested_resolution='Consider standardizing or removing these values'
            )
        return None


class SimilarValueDetector:
    """Detects values that are nearly identical (e.g., 'USA' vs 'United States')."""
    
    def __init__(self, similarity_threshold: float = 0.85, min_frequency: int = 1):
        """
        Args:
            similarity_threshold: String similarity threshold (0-1)
            min_frequency: Only consider values appearing at least this many times
        """
        self.similarity_threshold = similarity_threshold
        self.min_frequency = min_frequency
    
    def _normalize_for_comparison(self, s: str) -> str:
        """Normalize string for comparison (lowercase, stripped)."""
        return s.lower().strip()
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        norm1 = self._normalize_for_comparison(s1)
        norm2 = self._normalize_for_comparison(s2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def detect(self, values: List[str]) -> List[CategoryIssue]:
        """Find groups of similar values."""
        # Count frequencies
        freq = Counter(values)
        all_values = [v for v, c in freq.items() if c >= self.min_frequency]
        
        issues = []
        checked = set()
        
        for i, val1 in enumerate(all_values):
            if val1 in checked:
                continue
            
            similar_group = [val1]
            for val2 in all_values[i+1:]:
                if val2 not in checked:
                    sim = self._similarity(val1, val2)
                    if sim >= self.similarity_threshold:
                        similar_group.append(val2)
                        checked.add(val2)
            
            if len(similar_group) > 1:
                # Pick most common as canonical
                canonical = max(similar_group, key=lambda v: freq[v])
                
                issues.append(CategoryIssue(
                    issue_type='similar',
                    affected_values=similar_group,
                    suggested_resolution=canonical,
                    confidence=min(self._similarity(canonical, v) for v in similar_group)
                ))
                checked.add(val1)
        
        return issues


class SpellingDetector:
    """Detects potential spelling mistakes in categorical values."""
    
    def __init__(self, edit_distance_threshold: int = 2, min_frequency_ratio: float = 2.0):
        """
        Args:
            edit_distance_threshold: Max character edits to consider similar
            min_frequency_ratio: Rare value must be X times less common than frequent value
        """
        self.edit_distance_threshold = edit_distance_threshold
        self.min_frequency_ratio = min_frequency_ratio
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr
        
        return prev[-1]
    
    def detect(self, values: List[str]) -> List[CategoryIssue]:
        """Find potential spelling mistakes."""
        freq = Counter(values)
        sorted_values = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        issues = []
        checked = set()
        
        for rare_val, rare_count in sorted_values:
            if rare_val in checked:
                continue
            
            rare_val_lower = rare_val.lower().strip()
            
            for common_val, common_count in sorted_values:
                if common_val == rare_val or common_val in checked:
                    continue
                
                common_val_lower = common_val.lower().strip()
                
                # Skip if they're the same after normalization (case difference only)
                if rare_val_lower == common_val_lower:
                    continue
                
                # Check if rare value is similar to a more common value
                if common_count >= rare_count * self.min_frequency_ratio or (
                    common_count > rare_count and rare_count == 1
                ):
                    distance = self._levenshtein_distance(rare_val_lower, common_val_lower)
                    
                    if distance <= self.edit_distance_threshold:
                        issues.append(CategoryIssue(
                            issue_type='spelling',
                            affected_values=[rare_val],
                            suggested_resolution=common_val,
                            confidence=1.0 - (distance / max(len(rare_val), len(common_val)))
                        ))
                        checked.add(rare_val)
                        break
        
        return issues


class NegationPairDetector:
    """Detects negation pairs like 'active'/'inactive' that should be kept separate."""
    
    def __init__(self):
        self.negation_prefixes = ['in', 'un', 'non', 'dis', 'im']
    
    def detect(self, values: List[str]) -> Optional[CategoryIssue]:
        """Find negation pairs that should not be merged."""
        value_set_lower = {v.lower().strip(): v for v in set(values)}
        
        pairs = []
        for val_lower, val_orig in value_set_lower.items():
            # Check if negation exists
            for prefix in self.negation_prefixes:
                if val_lower.startswith(prefix):
                    # Check if base word exists
                    base_word = val_lower[len(prefix):]
                    if base_word in value_set_lower:
                        pairs.append((value_set_lower[base_word], val_orig))
        
        if pairs:
            return CategoryIssue(
                issue_type='negation_pair',
                affected_values=[item for pair in pairs for item in pair],
                suggested_resolution='Keep these separate (they are negation pairs)',
                metadata={'pairs': pairs}
            )
        return None


class CaseVariationDetector:
    """Detects values that differ only in case."""
    
    def detect(self, values: List[str]) -> List[CategoryIssue]:
        """Find case variations."""
        # Group by lowercase version
        case_groups = {}
        for val in values:
            lower_val = val.lower().strip()
            if lower_val not in case_groups:
                case_groups[lower_val] = []
            case_groups[lower_val].append(val)
        
        issues = []
        for lower_val, orig_vals in case_groups.items():
            unique_forms = list(set(orig_vals))
            if len(unique_forms) > 1:
                # Pick most common form as canonical
                counter = Counter(orig_vals)
                canonical = counter.most_common(1)[0][0]
                
                issues.append(CategoryIssue(
                    issue_type='case_variation',
                    affected_values=unique_forms,
                    suggested_resolution=canonical,
                    confidence=1.0
                ))
        
        return issues


# ============================================================================
# RESOLVERS
# ============================================================================

class SimilarValueResolver:
    """Resolves similar values to a canonical form."""
    
    def resolve(self, issue: CategoryIssue) -> List[ResolutionMapping]:
        """Map similar values to canonical value."""
        canonical = issue.suggested_resolution
        return [
            ResolutionMapping(
                original=val,
                resolved=canonical,
                reason=f"Standardized similar value (confidence: {issue.confidence:.2%})"
            )
            for val in issue.affected_values if val != canonical
        ]


class SpellingResolver:
    """Resolves spelling mistakes to correct form."""
    
    def resolve(self, issue: CategoryIssue) -> List[ResolutionMapping]:
        """Map misspelled values to correct spelling."""
        return [
            ResolutionMapping(
                original=issue.affected_values[0],
                resolved=issue.suggested_resolution,
                reason=f"Corrected spelling (confidence: {issue.confidence:.2%})"
            )
        ]


class PlaceholderResolver:
    """Resolves placeholder values."""
    
    def __init__(self, replacement: Optional[str] = None):
        """
        Args:
            replacement: Value to replace placeholders with (None = keep as is)
        """
        self.replacement = replacement
    
    def resolve(self, issue: CategoryIssue) -> List[ResolutionMapping]:
        """Standardize placeholder values."""
        if self.replacement is None:
            return []
        
        return [
            ResolutionMapping(
                original=val,
                resolved=self.replacement,
                reason="Standardized placeholder value"
            )
            for val in issue.affected_values
        ]


class CaseVariationResolver:
    """Resolves case variations."""
    
    def resolve(self, issue: CategoryIssue) -> List[ResolutionMapping]:
        """Map case variations to canonical form."""
        canonical = issue.suggested_resolution
        return [
            ResolutionMapping(
                original=val,
                resolved=canonical,
                reason="Standardized case"
            )
            for val in issue.affected_values if val != canonical
        ]


# ============================================================================
# ORCHESTRATORS
# ============================================================================

class CategoryDetector:
    """Orchestrates multiple detectors to find all issues."""
    
    def __init__(self):
        self.detectors = []
    
    def add_detector(self, detector):
        """Add a detector to the pipeline."""
        self.detectors.append(detector)
        return self
    
    def detect_issues(self, values: List[str]) -> List[CategoryIssue]:
        """Run all detectors and collect issues."""
        all_issues = []
        
        for detector in self.detectors:
            result = detector.detect(values)
            if result:
                if isinstance(result, list):
                    all_issues.extend(result)
                else:
                    all_issues.append(result)
        
        return all_issues


class CategoryResolver:
    """Orchestrates multiple resolvers to fix issues."""
    
    def __init__(self):
        self.resolvers = {
            'similar': SimilarValueResolver(),
            'spelling': SpellingResolver(),
            'placeholder': PlaceholderResolver(),
            'case_variation': CaseVariationResolver()
        }
        self.skip_types = {'high_cardinality', 'negation_pair'}
    
    def add_resolver(self, issue_type: str, resolver):
        """Add or replace a resolver for an issue type."""
        self.resolvers[issue_type] = resolver
        return self
    
    def resolve_issues(self, issues: List[CategoryIssue]) -> Dict[str, str]:
        """Convert issues to a mapping of original -> resolved values."""
        mapping = {}
        
        for issue in issues:
            # Skip certain issue types
            if issue.issue_type in self.skip_types:
                continue
                
            if issue.issue_type in self.resolvers:
                resolver = self.resolvers[issue.issue_type]
                resolutions = resolver.resolve(issue)
                
                for res in resolutions:
                    mapping[res.original] = res.resolved
        
        return mapping
    
    def apply_mapping(self, values: List[str], mapping: Dict[str, str]) -> List[str]:
        """Apply resolution mapping to values."""
        return [mapping.get(v, v) for v in values]


# ============================================================================
# PANDAS INTEGRATION
# ============================================================================

import pandas as pd


def analyze_categorical_column(
    df: pd.DataFrame, 
    column: str,
    uniqueness_threshold: float = 0.95,
    min_samples_for_categorical: int = 10,
    similarity_threshold: float = 0.85,
    edit_distance_threshold: int = 2,
    custom_placeholders: Optional[Set[str]] = None,
    placeholder_replacement: Optional[str] = None,
    detect_negation_pairs: bool = True,
    detect_case_variations: bool = True
) -> Tuple[List[CategoryIssue], Dict[str, str], bool]:
    """
    Analyze a single categorical column for issues.
    
    Returns:
        Tuple of (issues found, resolution mapping, is_truly_categorical)
    """
    # Get column values as list (handle NaN)
    values = df[column].fillna('').astype(str).tolist()
    
    # Setup detector
    detector = CategoryDetector()
    
    # High cardinality check FIRST
    high_card_detector = HighCardinalityDetector(
        uniqueness_threshold=uniqueness_threshold,
        min_samples=min_samples_for_categorical
    )
    detector.add_detector(high_card_detector)
    
    # Check if it's truly categorical
    high_card_issue = high_card_detector.detect(values)
    is_truly_categorical = high_card_issue is None
    
    if not is_truly_categorical:
        # Return early if not categorical
        return ([high_card_issue], {}, False)
    
    # Add other detectors only if it's categorical
    if detect_case_variations:
        detector.add_detector(CaseVariationDetector())
    
    if detect_negation_pairs:
        detector.add_detector(NegationPairDetector())
    
    detector.add_detector(PlaceholderDetector(custom_placeholders=custom_placeholders))
    detector.add_detector(SimilarValueDetector(similarity_threshold=similarity_threshold))
    detector.add_detector(SpellingDetector(edit_distance_threshold=edit_distance_threshold))
    
    # Detect issues
    issues = detector.detect_issues(values)
    
    # Setup resolver
    resolver = CategoryResolver()
    resolver.add_resolver('placeholder', PlaceholderResolver(replacement=placeholder_replacement))
    
    # Get resolution mapping
    mapping = resolver.resolve_issues(issues)
    
    return issues, mapping, is_truly_categorical


def clean_categorical_column(
    df: pd.DataFrame, 
    column: str, 
    mapping: Dict[str, str],
    inplace: bool = False
) -> pd.DataFrame:
    """Apply resolution mapping to clean a categorical column."""
    if not inplace:
        df = df.copy()
    
    df[column] = df[column].astype(str).replace(mapping)
    
    return df


def print_analysis_report(column: str, issues: List[CategoryIssue], mapping: Dict[str, str], is_categorical: bool):
    """Print a formatted report of issues and resolutions for a column."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT: {column}")
    print(f"{'='*80}")
    
    if not is_categorical:
        print("⚠️  Column is NOT truly categorical (too many unique values)")
        print("   Skipping further analysis.")
        return
    
    if not issues:
        print("✓ No issues detected")
        return
    
    # Group issues by type
    issue_groups = {}
    for issue in issues:
        if issue.issue_type not in issue_groups:
            issue_groups[issue.issue_type] = []
        issue_groups[issue.issue_type].append(issue)
    
    for issue_type, type_issues in issue_groups.items():
        print(f"\n{issue_type.upper().replace('_', ' ')}:")
        for issue in type_issues:
            if issue.issue_type == 'negation_pair':
                pairs = issue.metadata.get('pairs', [])
                for pos, neg in pairs:
                    print(f"   {pos} ↔ {neg}")
            else:
                print(f"   Affected: {issue.affected_values}")
                if issue.suggested_resolution:
                    print(f"   → {issue.suggested_resolution}")
    
    if mapping:
        print(f"\n{'─'*80}")
        print("RESOLUTIONS TO BE APPLIED:")
        print(f"{'─'*80}")
        for orig, resolved in mapping.items():
            print(f"  {orig:30} → {resolved}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Your actual data scenario
    df = pd.DataFrame({
        'country': ['Error', 'UK', 'USA', 'United Kingdom', 'United States', 'Unknown', 'usa'] * 2,
        'status': ['Active', 'Error', 'N/A', 'active', 'actve', 'inactive'] * 2 + ['Active', 'inactive'],
        'user_id': ['user_0', 'user_1', 'user_2', 'user_3', 'user_4', 
                    'user_5', 'user_6', 'user_7', 'user_8', 'user_9', 
                    'user_0', 'user_1', 'user_2', 'user_3']
    })
    
    print("TESTING WITH YOUR ACTUAL DATA:")
    print("=" * 80)
    print(df)
    
    for col in ['country', 'status', 'user_id']:
        issues, mapping, is_categorical = analyze_categorical_column(
            df, col,
            uniqueness_threshold=0.9,
            similarity_threshold=0.80
        )
        
        print_analysis_report(col, issues, mapping, is_categorical)
        
        if is_categorical and mapping:
            df_cleaned = clean_categorical_column(df, col, mapping)
            print(f"\nBefore: {sorted(df[col].unique())}")
            print(f"After:  {sorted(df_cleaned[col].unique())}")