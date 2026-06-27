# Feature Inconsistency Agent

Detects and resolves inconsistent values in tabular datasets — both numeric (impossible values) and categorical (surface variants like typos, abbreviations, and mixed casing).

The agent is built around a strategy pattern: each detection method is an independent, stateless unit. The orchestrator wires them together, applies changes to the dataframe, and records results — nothing more.

---

## Approach

**Numeric columns** are checked against registered min/max rules first. If no rules exist, an LLM inspects a random sample and flags logically impossible values. Confirmed issues are NaN-replaced.

**Categorical columns** go through a three-stage pipeline:
1. Edit-distance graph clustering groups similar unique values.
2. Auto-resolvable clusters (clear canonical form) are mapped directly — no LLM involved.
3. Ambiguous clusters are sent to the LLM in a **single batched call** per column. Every mapping is gated by a ValidationLayer before it touches the dataframe.

---

## File Map

```
feature_inconsistency_agent/
│
├── data_standardizing_service.py       # Orchestrator. Routes columns to detectors,
│                                       # applies changes, forwards results to collector.
│
├── detection_strategy.py               # Abstract base + shared types.
│                                       # DetectionStrategy, DetectionResult,
│                                       # ColumnIssue, DetectionContext, FallbackDetector.
│
├── numeric_detectors.py                # Two DetectionStrategy implementations:
│                                       # NumericRuleDetector (min/max, no LLM)
│                                       # NumericLLMDetector (LLM fallback, random sample)
│
├── categorical_cluster_detector.py     # DetectionStrategy for categorical columns.
│                                       # Runs cluster pipeline, delegates resolution
│                                       # to ClusterResolver, returns ColumnIssue list.
│
├── cluster_resolver.py                 # Resolves clusters into a {variant: canonical}
│                                       # mapping. Auto clusters mapped directly;
│                                       # ambiguous clusters resolved in one LLM call.
│
├── value_clusterer.py                  # Edit-distance graph clustering.
│                                       # Produces Cluster objects from unique values.
│
├── validation_layer.py                 # Domain rules: allowed values, min/max bounds.
│                                       # Gates every mapping before it is applied.
│
├── llm_client.py                       # Abstract LLMClient interface + ChatMessage /
│                                       # ChatResponse value objects. Provider-agnostic.
│
├── groq_llm_client.py                  # Groq implementation of LLMClient.
│                                       # Handles retries and delegates rate limiting
│                                       # to RateLimiter.
│
├── rate_limiter.py                     # Sliding-window rate limiter (RPM + TPM).
│                                       # Shared across agents hitting the same account.
│
├── result_collector.py                 # Typed, append-only result store.
│                                       # Owns all pipeline output — nothing else writes
│                                       # to a results dict directly.
│
├── standardization_evaluator.py        # Offline evaluator. Compares accepted mappings
│                                       # against ground truth. Returns precision /
│                                       # recall / F1 per column and overall.
│
└── llm_json_parser.py                  # Utility. Strips markdown fences and extracts
                                        # JSON from LLM responses. Used by all modules
                                        # that call the LLM.
```

---

## Key Design Decisions

**Strategy pattern for detection**
Adding a new detection method means implementing `DetectionStrategy` — one class, one method. The orchestrator and everything else stays untouched.

**LLM calls minimized**
Categorical columns make at most one LLM call regardless of how many unique values they have. Numeric columns only call the LLM when no rules are registered.

**Provider-agnostic LLM interface**
`LLMClient` is an abstract interface. Swapping Groq for another provider, or injecting a mock in tests, requires no changes outside `groq_llm_client.py`.

**Detectors are stateless**
Detector instances are created once and reused across all columns. No state leaks between runs.

**Validation always gates changes**
The `ValidationLayer` is the last check before any value is written to the dataframe — LLM decisions cannot bypass it.

---

## Quick Start

```python
from groq import Groq
from rate_limiter import RateLimiter
from groq_llm_client import GroqLLMClient
from validation_layer import ValidationLayer
from data_standardizing_service import DataStandardizingService

llm_client = GroqLLMClient(
    groq_client=Groq(api_key="..."),
    model="llama3-8b-8192",
    rate_limiter=RateLimiter(requests_per_minute=20, tokens_per_minute=30_000),
)

validation = ValidationLayer()
validation.register("age", min_value=0, max_value=120)
validation.register("gender", allowed_values=["Male", "Female", "Other"])

svc = DataStandardizingService(df=df, llm_client=llm_client, validation=validation)
results = svc.standardize()
```

**Evaluating against ground truth:**

```python
from standardization_evaluator import StandardizationEvaluator

evaluator = StandardizationEvaluator(original_df=svc.original_df, collector=svc._collector)
report = evaluator.evaluate({"gender": {"male": "Male", "M": "Male", "female": "Female"}})
evaluator.print_report(report)
```